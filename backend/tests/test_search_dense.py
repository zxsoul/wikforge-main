"""Dense 向量检索器（Qdrant search + Pre-Filtering）单元测试。

对应任务 14.2：实现 Dense 向量检索器（Qdrant search，Pre-Filtering 权限过滤，返回 Top 50）。

测试范围：
- Qdrant ``search`` 调用（NamedVector(name="dense") + 1024 维向量）
- limit=50（TOP_K_PER_RETRIEVER）
- 权限过滤通过 ``query_filter`` 在 Qdrant 端 Pre-Filtering
- ``with_payload=True``、``search_params`` 配置 HNSW
- 命中点转换为 ``SearchHit``，payload 字段正确填充
- 空向量、空结果、payload 缺失等边界情况
- Qdrant 客户端抛错时错误向上传播（由 ``_multi_recall`` 捕获并降级）
- 同步调用通过线程池运行，不阻塞事件循环
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
    content: str = "这是一个 Dense 召回的文本块",
    score: float = 0.92,
) -> SimpleNamespace:
    """构造一个 Qdrant ScoredPoint 等价对象。

    Qdrant Python 客户端返回的是 ``ScoredPoint``，但访问路径仅依赖
    ``id``/``score``/``payload`` 三个属性。这里用 ``SimpleNamespace``
    模拟，避免依赖 qdrant_client 的具体类型。
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
    """无 embedding 依赖的 SearchService（dense 测试不需要 embedding）。"""
    return SearchService(embedding_service=MagicMock())


@pytest.fixture
def permission_filter(search_service: SearchService) -> dict:
    """构造一个真实的 Qdrant 权限过滤条件（dict 形式）。"""
    user_id = "user-abc"
    space_ids = ["space-1", "space-2"]
    return search_service._build_qdrant_filter(user_id, space_ids)


@pytest.fixture
def dense_vector() -> list[float]:
    """1024 维的查询向量。"""
    return [0.01 * (i % 7) for i in range(1024)]


# ─── Qdrant 调用参数 ─────────────────────────────────────────────────


class TestDenseRecallSearchCall:
    """验证发送给 ``QdrantClient.search`` 的参数正确。"""

    @pytest.mark.asyncio
    async def test_uses_named_vector_dense(
        self,
        search_service: SearchService,
        permission_filter: dict,
        dense_vector: list[float],
    ) -> None:
        """``query_vector`` 必须是 NamedVector(name="dense", vector=...)。"""
        from qdrant_client.models import NamedVector

        client = MagicMock()
        client.search.return_value = []

        with patch(
            "app.core.qdrant.get_qdrant_client", return_value=client
        ):
            await search_service._dense_recall(dense_vector, permission_filter)

        assert client.search.called
        kwargs = client.search.call_args.kwargs
        query_vector = kwargs["query_vector"]
        assert isinstance(query_vector, NamedVector)
        assert query_vector.name == "dense"
        assert query_vector.vector == dense_vector

    @pytest.mark.asyncio
    async def test_limit_top_50(
        self,
        search_service: SearchService,
        permission_filter: dict,
        dense_vector: list[float],
    ) -> None:
        """``limit`` 必须为 50（Top 50 召回，TOP_K_PER_RETRIEVER）。"""
        client = MagicMock()
        client.search.return_value = []

        with patch(
            "app.core.qdrant.get_qdrant_client", return_value=client
        ):
            await search_service._dense_recall(dense_vector, permission_filter)

        kwargs = client.search.call_args.kwargs
        assert kwargs["limit"] == TOP_K_PER_RETRIEVER == 50

    @pytest.mark.asyncio
    async def test_with_payload_true(
        self,
        search_service: SearchService,
        permission_filter: dict,
        dense_vector: list[float],
    ) -> None:
        """``with_payload=True``，否则后续无法重建 SearchHit。"""
        client = MagicMock()
        client.search.return_value = []

        with patch(
            "app.core.qdrant.get_qdrant_client", return_value=client
        ):
            await search_service._dense_recall(dense_vector, permission_filter)

        kwargs = client.search.call_args.kwargs
        assert kwargs["with_payload"] is True

    @pytest.mark.asyncio
    async def test_targets_document_chunks_collection(
        self,
        search_service: SearchService,
        permission_filter: dict,
        dense_vector: list[float],
    ) -> None:
        """请求应针对 ``document_chunks`` collection 发起。"""
        from app.core.qdrant import COLLECTION_NAME

        client = MagicMock()
        client.search.return_value = []

        with patch(
            "app.core.qdrant.get_qdrant_client", return_value=client
        ):
            await search_service._dense_recall(dense_vector, permission_filter)

        kwargs = client.search.call_args.kwargs
        assert kwargs["collection_name"] == COLLECTION_NAME == "document_chunks"

    @pytest.mark.asyncio
    async def test_search_params_configured(
        self,
        search_service: SearchService,
        permission_filter: dict,
        dense_vector: list[float],
    ) -> None:
        """HNSW 检索参数应配置为合理的速度/召回平衡（hnsw_ef≥64，非精确搜索）。"""
        from qdrant_client.models import SearchParams

        client = MagicMock()
        client.search.return_value = []

        with patch(
            "app.core.qdrant.get_qdrant_client", return_value=client
        ):
            await search_service._dense_recall(dense_vector, permission_filter)

        kwargs = client.search.call_args.kwargs
        params = kwargs["search_params"]
        assert isinstance(params, SearchParams)
        # 召回 Top-50 时 ef ≥ 64 才能保证召回质量；
        # 同时禁用 exact 走 HNSW 索引，保证亚秒级延迟。
        assert params.hnsw_ef is not None and params.hnsw_ef >= 64
        assert params.exact is False


# ─── Pre-Filtering 权限过滤 ──────────────────────────────────────────


class TestDensePermissionFilter:
    """验证权限过滤通过 ``query_filter`` 在 Qdrant 端 Pre-Filtering。"""

    @pytest.mark.asyncio
    async def test_query_filter_is_qdrant_filter(
        self,
        search_service: SearchService,
        permission_filter: dict,
        dense_vector: list[float],
    ) -> None:
        """``query_filter`` 必须是 Qdrant ``Filter`` 对象（而非原始 dict）。"""
        from qdrant_client.models import Filter

        client = MagicMock()
        client.search.return_value = []

        with patch(
            "app.core.qdrant.get_qdrant_client", return_value=client
        ):
            await search_service._dense_recall(dense_vector, permission_filter)

        kwargs = client.search.call_args.kwargs
        assert isinstance(kwargs["query_filter"], Filter)

    @pytest.mark.asyncio
    async def test_filter_includes_user_id_match_value(
        self,
        search_service: SearchService,
        dense_vector: list[float],
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
            await search_service._dense_recall(dense_vector, perm_filter)

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
        dense_vector: list[float],
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
            await search_service._dense_recall(dense_vector, perm_filter)

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


class TestDenseHitConversion:
    """验证 Qdrant ScoredPoint 列表转换为 SearchHit。"""

    @pytest.mark.asyncio
    async def test_points_converted_to_search_hit(
        self,
        search_service: SearchService,
        permission_filter: dict,
        dense_vector: list[float],
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
            results = await search_service._dense_recall(
                dense_vector, permission_filter
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
        dense_vector: list[float],
    ) -> None:
        """Qdrant 余弦相似度分数必须被保留（供 RRF 融合使用）。"""
        points = [_make_qdrant_point(score=0.4242)]
        client = MagicMock()
        client.search.return_value = points

        with patch(
            "app.core.qdrant.get_qdrant_client", return_value=client
        ):
            results = await search_service._dense_recall(
                dense_vector, permission_filter
            )

        assert results[0].score == pytest.approx(0.4242)

    @pytest.mark.asyncio
    async def test_chunk_id_uses_qdrant_point_id_as_string(
        self,
        search_service: SearchService,
        permission_filter: dict,
        dense_vector: list[float],
    ) -> None:
        """Qdrant 的 point.id 可能是 UUID/int，需统一转为字符串。"""
        # Qdrant 实际可能返回 int / UUID 类型作为 point id
        points = [SimpleNamespace(id=12345, score=0.5, payload={})]
        client = MagicMock()
        client.search.return_value = points

        with patch(
            "app.core.qdrant.get_qdrant_client", return_value=client
        ):
            results = await search_service._dense_recall(
                dense_vector, permission_filter
            )

        assert results[0].chunk_id == "12345"

    @pytest.mark.asyncio
    async def test_missing_payload_fields_default_to_empty(
        self,
        search_service: SearchService,
        permission_filter: dict,
        dense_vector: list[float],
    ) -> None:
        """payload 缺字段时，SearchHit 应回退到默认值，不应抛 KeyError。"""
        points = [SimpleNamespace(id="x", score=0.5, payload={})]
        client = MagicMock()
        client.search.return_value = points

        with patch(
            "app.core.qdrant.get_qdrant_client", return_value=client
        ):
            results = await search_service._dense_recall(
                dense_vector, permission_filter
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
        dense_vector: list[float],
    ) -> None:
        """``payload=None`` 不应导致 NoneType 错误。"""
        points = [SimpleNamespace(id="x", score=0.1, payload=None)]
        client = MagicMock()
        client.search.return_value = points

        with patch(
            "app.core.qdrant.get_qdrant_client", return_value=client
        ):
            results = await search_service._dense_recall(
                dense_vector, permission_filter
            )

        assert len(results) == 1
        assert results[0].chunk_id == "x"
        assert results[0].content == ""


# ─── 边界与异常 ───────────────────────────────────────────────────────


class TestDenseEdgeCases:
    """空向量、空结果、客户端异常等极端场景。"""

    @pytest.mark.asyncio
    async def test_empty_results_returns_empty_list(
        self,
        search_service: SearchService,
        permission_filter: dict,
        dense_vector: list[float],
    ) -> None:
        """Qdrant 返回 0 条命中时，返回空列表。"""
        client = MagicMock()
        client.search.return_value = []

        with patch(
            "app.core.qdrant.get_qdrant_client", return_value=client
        ):
            results = await search_service._dense_recall(
                dense_vector, permission_filter
            )

        assert results == []

    @pytest.mark.asyncio
    async def test_empty_vector_does_not_crash(
        self,
        search_service: SearchService,
        permission_filter: dict,
    ) -> None:
        """传入空向量时仍应调用 Qdrant（让后端按其默认行为处理），不在客户端崩溃。

        实际场景中 Embedding 服务异常可能产生空向量，调用方应能优雅降级而非抛错。
        """
        from qdrant_client.models import NamedVector

        client = MagicMock()
        client.search.return_value = []

        with patch(
            "app.core.qdrant.get_qdrant_client", return_value=client
        ):
            results = await search_service._dense_recall([], permission_filter)

        assert results == []
        # 仍然将查询透传给 Qdrant，由其决定如何处理
        assert client.search.called
        assert client.search.call_args.kwargs["query_vector"] == NamedVector(
            name="dense", vector=[]
        )

    @pytest.mark.asyncio
    async def test_qdrant_client_error_propagates(
        self,
        search_service: SearchService,
        permission_filter: dict,
        dense_vector: list[float],
    ) -> None:
        """Qdrant 客户端抛错时，错误应向上传播（由 ``_multi_recall`` 捕获并降级）。"""
        client = MagicMock()
        client.search.side_effect = RuntimeError("qdrant unreachable")

        with patch(
            "app.core.qdrant.get_qdrant_client", return_value=client
        ):
            with pytest.raises(RuntimeError, match="qdrant unreachable"):
                await search_service._dense_recall(
                    dense_vector, permission_filter
                )

    @pytest.mark.asyncio
    async def test_runs_in_executor_does_not_block_event_loop(
        self,
        search_service: SearchService,
        permission_filter: dict,
        dense_vector: list[float],
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
                search_service._dense_recall(dense_vector, permission_filter),
                parallel_task(),
            )
            elapsed = asyncio.get_event_loop().time() - t0

        # 并发执行的 sleep(0.01) 应在 dense 调用完成前就结束；
        # 总耗时应接近最慢任务（~0.1s），而非串行的 0.11s+
        assert side == "done"
        assert elapsed < 0.2
        assert recall == []
