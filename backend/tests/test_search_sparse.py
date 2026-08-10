"""Sparse 向量检索器（Qdrant sparse search + Pre-Filtering）单元测试。

对应任务 14.3：实现 Sparse 向量检索器（Qdrant sparse search，Pre-Filtering 权限过滤，返回 Top 50）。

测试范围：
- Qdrant ``search`` 调用使用 ``NamedSparseVector(name="sparse", vector=SparseVector(...))``
  携带传入的 ``indices`` / ``values``
- ``limit=50`` (TOP_K_PER_RETRIEVER)
- ``with_payload=True``
- 权限过滤通过 ``query_filter`` 在 Qdrant 端 Pre-Filtering（``allowed_user_ids`` 精确匹配 +
  ``space_id`` MatchAny）
- ``sparse_indices`` 为空时直接返回 ``[]``，不发起 Qdrant 调用
- 命中点转换为 ``SearchHit``，payload 字段正确填充
- 空结果、payload 缺失/为 None、point.id 非字符串等边界情况
- Qdrant 客户端抛错时错误向上传播（由 ``_multi_recall`` 捕获并降级）
- 同步调用通过线程池运行，不阻塞事件循环

风格与 ``test_search_dense.py`` 保持一致。
"""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.services.search_service import (
    TOP_K_PER_RETRIEVER,
    SearchHit,
    SearchService,
)


# ─── Helpers ──────────────────────────────────────────────────────────


def _make_qdrant_point(
    *,
    point_id: str | None = None,
    document_id: str | None = None,
    space_id: str | None = None,
    chunk_index: int = 0,
    title_chain: str = "Section > Subsection",
    source_file: str = "test.pdf",
    content: str = "这是一个 Sparse 召回的文本块",
    score: float = 0.88,
) -> SimpleNamespace:
    """构造一个 Qdrant ScoredPoint 等价对象。

    与 dense 测试相同：仅依赖 ``id``/``score``/``payload`` 三个属性，
    使用 ``SimpleNamespace`` 模拟即可。
    """
    return SimpleNamespace(
        id=point_id or str(uuid.uuid4()),
        score=score,
        payload={
            "document_id": document_id or str(uuid.uuid4()),
            "space_id": space_id or str(uuid.uuid4()),
            "chunk_index": chunk_index,
            "title_chain": title_chain,
            "source_file": source_file,
            "content": content,
        },
    )


@pytest.fixture
def search_service() -> SearchService:
    """无 embedding 依赖的 SearchService（sparse 测试不需要 embedding）。"""
    return SearchService(embedding_service=MagicMock())


@pytest.fixture
def permission_filter(search_service: SearchService) -> dict:
    """构造一个真实的 Qdrant 权限过滤条件（dict 形式）。"""
    user_id = "user-abc"
    space_ids = ["space-1", "space-2"]
    return search_service._build_qdrant_filter(user_id, space_ids)


@pytest.fixture
def sparse_indices() -> list[int]:
    """SPLADE 风格的稀疏向量索引（非零位置）。"""
    return [10, 42, 137, 2048, 9999]


@pytest.fixture
def sparse_values() -> list[float]:
    """对应 ``sparse_indices`` 的权重。"""
    return [0.91, 0.45, 0.31, 0.22, 0.18]


# ─── Qdrant 调用参数 ─────────────────────────────────────────────────


class TestSparseRecallSearchCall:
    """验证发送给 ``QdrantClient.search`` 的参数正确。"""

    @pytest.mark.asyncio
    async def test_uses_named_sparse_vector(
        self,
        search_service: SearchService,
        permission_filter: dict,
        sparse_indices: list[int],
        sparse_values: list[float],
    ) -> None:
        """``query_vector`` 必须是 NamedSparseVector(name="sparse", vector=SparseVector(...))。"""
        from qdrant_client.models import NamedSparseVector, SparseVector

        client = MagicMock()
        client.search.return_value = []

        with patch(
            "app.core.qdrant.get_qdrant_client", return_value=client
        ):
            await search_service._sparse_recall(
                sparse_indices, sparse_values, permission_filter
            )

        assert client.search.called
        kwargs = client.search.call_args.kwargs
        query_vector = kwargs["query_vector"]
        assert isinstance(query_vector, NamedSparseVector)
        assert query_vector.name == "sparse"
        assert isinstance(query_vector.vector, SparseVector)
        assert list(query_vector.vector.indices) == sparse_indices
        assert list(query_vector.vector.values) == sparse_values

    @pytest.mark.asyncio
    async def test_limit_top_50(
        self,
        search_service: SearchService,
        permission_filter: dict,
        sparse_indices: list[int],
        sparse_values: list[float],
    ) -> None:
        """``limit`` 必须为 50（Top 50 召回，TOP_K_PER_RETRIEVER）。"""
        client = MagicMock()
        client.search.return_value = []

        with patch(
            "app.core.qdrant.get_qdrant_client", return_value=client
        ):
            await search_service._sparse_recall(
                sparse_indices, sparse_values, permission_filter
            )

        kwargs = client.search.call_args.kwargs
        assert kwargs["limit"] == TOP_K_PER_RETRIEVER == 50

    @pytest.mark.asyncio
    async def test_with_payload_true(
        self,
        search_service: SearchService,
        permission_filter: dict,
        sparse_indices: list[int],
        sparse_values: list[float],
    ) -> None:
        """``with_payload=True``，否则后续无法重建 SearchHit。"""
        client = MagicMock()
        client.search.return_value = []

        with patch(
            "app.core.qdrant.get_qdrant_client", return_value=client
        ):
            await search_service._sparse_recall(
                sparse_indices, sparse_values, permission_filter
            )

        kwargs = client.search.call_args.kwargs
        assert kwargs["with_payload"] is True

    @pytest.mark.asyncio
    async def test_targets_document_chunks_collection(
        self,
        search_service: SearchService,
        permission_filter: dict,
        sparse_indices: list[int],
        sparse_values: list[float],
    ) -> None:
        """请求应针对 ``document_chunks`` collection 发起。"""
        from app.core.qdrant import COLLECTION_NAME

        client = MagicMock()
        client.search.return_value = []

        with patch(
            "app.core.qdrant.get_qdrant_client", return_value=client
        ):
            await search_service._sparse_recall(
                sparse_indices, sparse_values, permission_filter
            )

        kwargs = client.search.call_args.kwargs
        assert kwargs["collection_name"] == COLLECTION_NAME == "document_chunks"


# ─── Pre-Filtering 权限过滤 ──────────────────────────────────────────


class TestSparsePermissionFilter:
    """验证权限过滤通过 ``query_filter`` 在 Qdrant 端 Pre-Filtering。"""

    @pytest.mark.asyncio
    async def test_query_filter_is_qdrant_filter(
        self,
        search_service: SearchService,
        permission_filter: dict,
        sparse_indices: list[int],
        sparse_values: list[float],
    ) -> None:
        """``query_filter`` 必须是 Qdrant ``Filter`` 对象（而非原始 dict）。"""
        from qdrant_client.models import Filter

        client = MagicMock()
        client.search.return_value = []

        with patch(
            "app.core.qdrant.get_qdrant_client", return_value=client
        ):
            await search_service._sparse_recall(
                sparse_indices, sparse_values, permission_filter
            )

        kwargs = client.search.call_args.kwargs
        assert isinstance(kwargs["query_filter"], Filter)

    @pytest.mark.asyncio
    async def test_filter_includes_user_id_match_value(
        self,
        search_service: SearchService,
        sparse_indices: list[int],
        sparse_values: list[float],
    ) -> None:
        """权限过滤应包含 ``allowed_user_ids`` 的精确匹配。"""
        from qdrant_client.models import FieldCondition, MatchValue

        user_id = str(uuid.uuid4())
        perm_filter = search_service._build_qdrant_filter(
            user_id, ["space-1"]
        )
        client = MagicMock()
        client.search.return_value = []

        with patch(
            "app.core.qdrant.get_qdrant_client", return_value=client
        ):
            await search_service._sparse_recall(
                sparse_indices, sparse_values, perm_filter
            )

        qdrant_filter = client.search.call_args.kwargs["query_filter"]
        user_clauses = [
            c
            for c in qdrant_filter.should
            if isinstance(c, FieldCondition) and c.key == "allowed_user_ids"
        ]
        assert len(user_clauses) == 1
        assert isinstance(user_clauses[0].match, MatchValue)
        assert user_clauses[0].match.value == user_id

    @pytest.mark.asyncio
    async def test_filter_includes_space_ids_match_any(
        self,
        search_service: SearchService,
        sparse_indices: list[int],
        sparse_values: list[float],
    ) -> None:
        """权限过滤应通过 ``MatchAny`` 在 ``space_id`` 上匹配可访问空间集合。"""
        from qdrant_client.models import FieldCondition, MatchAny

        user_id = "user-1"
        space_ids = [str(uuid.uuid4()) for _ in range(3)]
        perm_filter = search_service._build_qdrant_filter(user_id, space_ids)

        client = MagicMock()
        client.search.return_value = []

        with patch(
            "app.core.qdrant.get_qdrant_client", return_value=client
        ):
            await search_service._sparse_recall(
                sparse_indices, sparse_values, perm_filter
            )

        qdrant_filter = client.search.call_args.kwargs["query_filter"]
        space_clauses = [
            c
            for c in qdrant_filter.should
            if isinstance(c, FieldCondition) and c.key == "space_id"
        ]
        assert len(space_clauses) == 1
        assert isinstance(space_clauses[0].match, MatchAny)
        assert space_clauses[0].match.any == space_ids


# ─── 命中转换 ─────────────────────────────────────────────────────────


class TestSparseHitConversion:
    """验证 Qdrant ScoredPoint 列表转换为 SearchHit。"""

    @pytest.mark.asyncio
    async def test_points_converted_to_search_hit(
        self,
        search_service: SearchService,
        permission_filter: dict,
        sparse_indices: list[int],
        sparse_values: list[float],
    ) -> None:
        """``client.search`` 返回的点应映射为 SearchHit 列表，字段一一对应。"""
        points = [
            _make_qdrant_point(
                point_id="c1",
                document_id="d1",
                space_id="s1",
                chunk_index=3,
                title_chain="一 > 1.1",
                source_file="规范.pdf",
                content="水泥工艺",
                score=0.93,
            ),
            _make_qdrant_point(
                point_id="c2",
                document_id="d1",
                space_id="s1",
                chunk_index=4,
                content="另一段",
                score=0.71,
            ),
        ]
        client = MagicMock()
        client.search.return_value = points

        with patch(
            "app.core.qdrant.get_qdrant_client", return_value=client
        ):
            results = await search_service._sparse_recall(
                sparse_indices, sparse_values, permission_filter
            )

        assert len(results) == 2
        first = results[0]
        assert isinstance(first, SearchHit)
        assert first.chunk_id == "c1"
        assert first.document_id == "d1"
        assert first.space_id == "s1"
        assert first.chunk_index == 3
        assert first.title_chain == "一 > 1.1"
        assert first.source_file == "规范.pdf"
        assert first.content == "水泥工艺"
        assert first.score == pytest.approx(0.93)

    @pytest.mark.asyncio
    async def test_score_preserved_from_qdrant(
        self,
        search_service: SearchService,
        permission_filter: dict,
        sparse_indices: list[int],
        sparse_values: list[float],
    ) -> None:
        """Qdrant sparse 相似度分数必须被保留（供 RRF 融合使用）。"""
        points = [_make_qdrant_point(score=0.6789)]
        client = MagicMock()
        client.search.return_value = points

        with patch(
            "app.core.qdrant.get_qdrant_client", return_value=client
        ):
            results = await search_service._sparse_recall(
                sparse_indices, sparse_values, permission_filter
            )

        assert results[0].score == pytest.approx(0.6789)

    @pytest.mark.asyncio
    async def test_chunk_id_uses_qdrant_point_id_as_string(
        self,
        search_service: SearchService,
        permission_filter: dict,
        sparse_indices: list[int],
        sparse_values: list[float],
    ) -> None:
        """Qdrant 的 point.id 可能是 UUID/int，需统一转为字符串。"""
        points = [SimpleNamespace(id=12345, score=0.5, payload={})]
        client = MagicMock()
        client.search.return_value = points

        with patch(
            "app.core.qdrant.get_qdrant_client", return_value=client
        ):
            results = await search_service._sparse_recall(
                sparse_indices, sparse_values, permission_filter
            )

        assert results[0].chunk_id == "12345"

    @pytest.mark.asyncio
    async def test_missing_payload_fields_default_to_empty(
        self,
        search_service: SearchService,
        permission_filter: dict,
        sparse_indices: list[int],
        sparse_values: list[float],
    ) -> None:
        """payload 缺字段时，SearchHit 应回退到默认值，不应抛 KeyError。"""
        points = [SimpleNamespace(id="x", score=0.5, payload={})]
        client = MagicMock()
        client.search.return_value = points

        with patch(
            "app.core.qdrant.get_qdrant_client", return_value=client
        ):
            results = await search_service._sparse_recall(
                sparse_indices, sparse_values, permission_filter
            )

        hit = results[0]
        assert hit.document_id == ""
        assert hit.space_id == ""
        assert hit.chunk_index == 0
        assert hit.title_chain == ""
        assert hit.source_file == ""
        assert hit.content == ""

    @pytest.mark.asyncio
    async def test_none_payload_handled(
        self,
        search_service: SearchService,
        permission_filter: dict,
        sparse_indices: list[int],
        sparse_values: list[float],
    ) -> None:
        """``payload=None`` 不应导致 NoneType 错误。"""
        points = [SimpleNamespace(id="x", score=0.1, payload=None)]
        client = MagicMock()
        client.search.return_value = points

        with patch(
            "app.core.qdrant.get_qdrant_client", return_value=client
        ):
            results = await search_service._sparse_recall(
                sparse_indices, sparse_values, permission_filter
            )

        assert len(results) == 1
        assert results[0].chunk_id == "x"
        assert results[0].content == ""


# ─── 边界与异常 ───────────────────────────────────────────────────────


class TestSparseEdgeCases:
    """空向量、空结果、客户端异常等极端场景。"""

    @pytest.mark.asyncio
    async def test_empty_sparse_indices_returns_empty_list_without_calling_qdrant(
        self,
        search_service: SearchService,
        permission_filter: dict,
    ) -> None:
        """``sparse_indices`` 为空时应直接返回 ``[]``，不发起 Qdrant 调用。

        Qdrant 不支持空 sparse 向量查询；同时为了节省一次 RPC，应在客户端短路。
        """
        client = MagicMock()
        client.search.return_value = []

        with patch(
            "app.core.qdrant.get_qdrant_client", return_value=client
        ):
            results = await search_service._sparse_recall(
                [], [], permission_filter
            )

        assert results == []
        assert not client.search.called

    @pytest.mark.asyncio
    async def test_empty_results_returns_empty_list(
        self,
        search_service: SearchService,
        permission_filter: dict,
        sparse_indices: list[int],
        sparse_values: list[float],
    ) -> None:
        """Qdrant 返回 0 条命中时，返回空列表。"""
        client = MagicMock()
        client.search.return_value = []

        with patch(
            "app.core.qdrant.get_qdrant_client", return_value=client
        ):
            results = await search_service._sparse_recall(
                sparse_indices, sparse_values, permission_filter
            )

        assert results == []

    @pytest.mark.asyncio
    async def test_returns_list_of_search_hit(
        self,
        search_service: SearchService,
        permission_filter: dict,
        sparse_indices: list[int],
        sparse_values: list[float],
    ) -> None:
        """返回值类型必须是 ``list[SearchHit]``。"""
        points = [_make_qdrant_point() for _ in range(5)]
        client = MagicMock()
        client.search.return_value = points

        with patch(
            "app.core.qdrant.get_qdrant_client", return_value=client
        ):
            results = await search_service._sparse_recall(
                sparse_indices, sparse_values, permission_filter
            )

        assert isinstance(results, list)
        assert len(results) == 5
        assert all(isinstance(hit, SearchHit) for hit in results)

    @pytest.mark.asyncio
    async def test_qdrant_client_error_propagates(
        self,
        search_service: SearchService,
        permission_filter: dict,
        sparse_indices: list[int],
        sparse_values: list[float],
    ) -> None:
        """Qdrant 客户端抛错时，错误应向上传播（由 ``_multi_recall`` 捕获并降级）。"""
        client = MagicMock()
        client.search.side_effect = RuntimeError("qdrant unreachable")

        with patch(
            "app.core.qdrant.get_qdrant_client", return_value=client
        ):
            with pytest.raises(RuntimeError, match="qdrant unreachable"):
                await search_service._sparse_recall(
                    sparse_indices, sparse_values, permission_filter
                )

    @pytest.mark.asyncio
    async def test_runs_in_executor_does_not_block_event_loop(
        self,
        search_service: SearchService,
        permission_filter: dict,
        sparse_indices: list[int],
        sparse_values: list[float],
    ) -> None:
        """同步 ``qdrant_client.search`` 调用必须放入线程池，避免阻塞事件循环。"""
        import time

        def slow_search(*args, **kwargs):
            time.sleep(0.1)
            return []

        client = MagicMock()
        client.search.side_effect = slow_search

        async def parallel_task() -> str:
            await asyncio.sleep(0.01)
            return "done"

        with patch(
            "app.core.qdrant.get_qdrant_client", return_value=client
        ):
            t0 = asyncio.get_event_loop().time()
            recall, side = await asyncio.gather(
                search_service._sparse_recall(
                    sparse_indices, sparse_values, permission_filter
                ),
                parallel_task(),
            )
            elapsed = asyncio.get_event_loop().time() - t0

        # 并发执行的 sleep(0.01) 应在 sparse 调用完成前就结束；
        # 总耗时应接近最慢任务（~0.1s），而非串行的 0.11s+
        assert side == "done"
        assert elapsed < 0.2
        assert recall == []
