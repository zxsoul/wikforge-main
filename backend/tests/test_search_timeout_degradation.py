"""搜索超时降级（任务 14.7）单元测试。

对应任务 14.7：实现搜索超时降级（单路 3 秒超时，跳过未返回的路）。
对应需求 6.6：IF 任一路召回在 3 秒内未返回结果, THEN THE Search_Engine SHALL
跳过该路召回，使用已返回的召回结果继续执行融合与精排流程。

设计参考：``Search Service - 多路并发召回``，使用 ``asyncio.wait_for`` 对
``_bm25_recall`` / ``_dense_recall`` / ``_sparse_recall`` 三路召回各自施加
3 秒超时，配合 ``asyncio.gather(..., return_exceptions=True)`` 收集结果，
异常（``asyncio.TimeoutError`` 与一般异常）被过滤后只把成功的列表返回给
RRF 融合环节。

本文件聚焦超时降级行为，与 ``test_search_bm25.py``/``test_search_dense.py``
/``test_search_sparse.py`` 中的"客户端抛错向上传播"测试互补。

测试矩阵：
- 单路超时：另外两路召回成功 → RRF 用两路结果融合
- 全部超时：返回空结果（``search()`` 顶层），不抛错
- 单路异常：被吞掉，其他两路成功
- 超时 + 异常混合：仅剩的一路成功
- 全部成功：三路并发返回，``_multi_recall`` 返回 3 个列表
- 并发性：慢路不应阻塞快路（总耗时不大于最慢路 + 少量调度开销）
- 超时窗口对齐设计：``RETRIEVER_TIMEOUT == 3.0``
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app.services.search_service as search_service_module
from app.services.search_service import (
    RETRIEVER_TIMEOUT,
    SearchHit,
    SearchResponse,
    SearchService,
)


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_embedding_service() -> AsyncMock:
    """模拟 EmbeddingService，返回固定 dense + sparse 向量。"""
    service = AsyncMock()
    service.embed_query = AsyncMock(
        return_value=MagicMock(
            dense_vector=[0.1] * 1024,
            sparse_indices=[1, 5, 10],
            sparse_values=[0.5, 0.3, 0.2],
        )
    )
    return service


@pytest.fixture
def search_service(mock_embedding_service: AsyncMock) -> SearchService:
    """注入 mock embedding 的 SearchService 实例。"""
    return SearchService(embedding_service=mock_embedding_service)


@pytest.fixture
def short_timeout(monkeypatch: pytest.MonkeyPatch) -> float:
    """把 RETRIEVER_TIMEOUT 缩短到 0.1 秒，避免测试串行等待 3 秒。

    模块内 ``_multi_recall`` 使用 ``asyncio.wait_for(task, timeout=RETRIEVER_TIMEOUT)``，
    通过 monkeypatch 设置模块级常量即可改变其行为。
    """
    monkeypatch.setattr(search_service_module, "RETRIEVER_TIMEOUT", 0.1)
    return 0.1


def _make_hit(chunk_id: str, content: str = "demo") -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        document_id=str(uuid.uuid4()),
        space_id=str(uuid.uuid4()),
        chunk_index=0,
        title_chain="Section > Sub",
        source_file="test.pdf",
        content=content,
        score=1.0,
    )


async def _make_slow_retriever(delay: float, chunk_id: str = "slow"):
    """构造一个会休眠 ``delay`` 秒后才返回的召回函数。"""

    async def _slow(*args, **kwargs):
        await asyncio.sleep(delay)
        return [_make_hit(chunk_id)]

    return _slow


async def _make_fast_retriever(chunk_id: str):
    """构造一个立刻返回单条命中的召回函数。"""

    async def _fast(*args, **kwargs):
        return [_make_hit(chunk_id)]

    return _fast


async def _make_failing_retriever(exc: Exception):
    """构造一个抛出指定异常的召回函数。"""

    async def _fail(*args, **kwargs):
        raise exc

    return _fail


# ─── Constant alignment ───────────────────────────────────────────────


class TestTimeoutConstant:
    """确认超时常量与设计/需求对齐。"""

    def test_retriever_timeout_is_three_seconds(self) -> None:
        """RETRIEVER_TIMEOUT 必须为 3.0 秒（需求 6.6）。"""
        assert RETRIEVER_TIMEOUT == 3.0


# ─── Single-path timeout ──────────────────────────────────────────────


class TestSinglePathTimeout:
    """单路超时场景：另外两路成功 → 仅这两路结果进入 RRF。"""

    @pytest.mark.asyncio
    async def test_bm25_times_out_dense_and_sparse_succeed(
        self, search_service: SearchService, short_timeout: float
    ) -> None:
        """BM25 超时，Dense 与 Sparse 在超时前返回 → 仅返回 2 路。"""
        slow = await _make_slow_retriever(delay=short_timeout * 10, chunk_id="bm25")
        fast_dense = await _make_fast_retriever("dense")
        fast_sparse = await _make_fast_retriever("sparse")

        with patch.object(
            search_service, "_bm25_recall", side_effect=slow
        ), patch.object(
            search_service, "_dense_recall", side_effect=fast_dense
        ), patch.object(
            search_service, "_sparse_recall", side_effect=fast_sparse
        ):
            results = await search_service._multi_recall(
                query="q",
                dense_vector=[0.1] * 1024,
                sparse_indices=[1],
                sparse_values=[0.5],
                qdrant_filter={},
                opensearch_filter={},
            )

        assert len(results) == 2
        chunk_ids = {hit.chunk_id for path in results for hit in path}
        assert chunk_ids == {"dense", "sparse"}
        assert "bm25" not in chunk_ids

    @pytest.mark.asyncio
    async def test_rrf_fusion_uses_only_successful_paths(
        self, search_service: SearchService, short_timeout: float
    ) -> None:
        """超时一路后，RRF 融合只对剩余两路结果做合并去重。"""
        slow = await _make_slow_retriever(delay=short_timeout * 10, chunk_id="bm25")
        fast_dense = await _make_fast_retriever("dense_only")
        fast_sparse = await _make_fast_retriever("sparse_only")

        with patch.object(
            search_service, "_bm25_recall", side_effect=slow
        ), patch.object(
            search_service, "_dense_recall", side_effect=fast_dense
        ), patch.object(
            search_service, "_sparse_recall", side_effect=fast_sparse
        ):
            recall_results = await search_service._multi_recall(
                query="q",
                dense_vector=[0.1] * 1024,
                sparse_indices=[1],
                sparse_values=[0.5],
                qdrant_filter={},
                opensearch_filter={},
            )
            candidates = search_service._rrf_fusion(recall_results)

        candidate_ids = {c.chunk_id for c in candidates}
        # 只有来自成功召回的两个 chunk 进入 RRF
        assert candidate_ids == {"dense_only", "sparse_only"}
        assert len(candidates) == 2


# ─── All-paths timeout ────────────────────────────────────────────────


class TestAllPathsTimeout:
    """三路全部超时场景：返回空结果，不抛异常。"""

    @pytest.mark.asyncio
    async def test_all_three_paths_time_out_returns_empty(
        self, search_service: SearchService, short_timeout: float
    ) -> None:
        """三路均超时 → ``_multi_recall`` 返回空列表。"""
        slow = await _make_slow_retriever(delay=short_timeout * 10)

        with patch.object(
            search_service, "_bm25_recall", side_effect=slow
        ), patch.object(
            search_service, "_dense_recall", side_effect=slow
        ), patch.object(
            search_service, "_sparse_recall", side_effect=slow
        ):
            results = await search_service._multi_recall(
                query="q",
                dense_vector=[0.1] * 1024,
                sparse_indices=[1],
                sparse_values=[0.5],
                qdrant_filter={},
                opensearch_filter={},
            )

        assert results == []

    @pytest.mark.asyncio
    async def test_all_paths_timeout_top_level_search_returns_empty_response(
        self, search_service: SearchService, short_timeout: float
    ) -> None:
        """端到端：三路全超时时 ``search()`` 返回空结果而非抛错。"""
        slow = await _make_slow_retriever(delay=short_timeout * 10)

        with patch.object(
            search_service, "_bm25_recall", side_effect=slow
        ), patch.object(
            search_service, "_dense_recall", side_effect=slow
        ), patch.object(
            search_service, "_sparse_recall", side_effect=slow
        ):
            response = await search_service.search(
                query="q",
                user_id="user-1",
                allowed_space_ids=["space-1"],
            )

        assert isinstance(response, SearchResponse)
        assert response.total == 0
        assert response.results == []


# ─── Exception handling ───────────────────────────────────────────────


class TestSinglePathException:
    """单路抛异常（非超时）场景：被吞掉，不影响其他两路。"""

    @pytest.mark.asyncio
    async def test_dense_raises_exception_other_paths_used(
        self, search_service: SearchService, short_timeout: float
    ) -> None:
        """Dense 抛 RuntimeError → BM25 + Sparse 仍参与 RRF。"""
        fast_bm25 = await _make_fast_retriever("bm25")
        failing = await _make_failing_retriever(RuntimeError("qdrant down"))
        fast_sparse = await _make_fast_retriever("sparse")

        with patch.object(
            search_service, "_bm25_recall", side_effect=fast_bm25
        ), patch.object(
            search_service, "_dense_recall", side_effect=failing
        ), patch.object(
            search_service, "_sparse_recall", side_effect=fast_sparse
        ):
            results = await search_service._multi_recall(
                query="q",
                dense_vector=[0.1] * 1024,
                sparse_indices=[1],
                sparse_values=[0.5],
                qdrant_filter={},
                opensearch_filter={},
            )

        assert len(results) == 2
        chunk_ids = {hit.chunk_id for path in results for hit in path}
        assert chunk_ids == {"bm25", "sparse"}

    @pytest.mark.asyncio
    async def test_unexpected_exception_does_not_raise(
        self, search_service: SearchService, short_timeout: float
    ) -> None:
        """召回抛 ValueError 等非超时异常时，``_multi_recall`` 不应向上抛。"""
        fast = await _make_fast_retriever("ok")
        failing = await _make_failing_retriever(ValueError("boom"))

        with patch.object(
            search_service, "_bm25_recall", side_effect=failing
        ), patch.object(
            search_service, "_dense_recall", side_effect=fast
        ), patch.object(
            search_service, "_sparse_recall", side_effect=fast
        ):
            # 不应抛
            results = await search_service._multi_recall(
                query="q",
                dense_vector=[0.1] * 1024,
                sparse_indices=[1],
                sparse_values=[0.5],
                qdrant_filter={},
                opensearch_filter={},
            )

        assert len(results) == 2


# ─── Mixed timeout + exception ────────────────────────────────────────


class TestMixedTimeoutAndException:
    """混合场景：一路超时 + 一路抛错 → 仅剩的一路成功。"""

    @pytest.mark.asyncio
    async def test_bm25_times_out_dense_raises_only_sparse_remains(
        self, search_service: SearchService, short_timeout: float
    ) -> None:
        """BM25 超时 + Dense 抛错 + Sparse 成功 → 仅 Sparse 进入 RRF。"""
        slow = await _make_slow_retriever(delay=short_timeout * 10)
        failing = await _make_failing_retriever(ConnectionError("qdrant timeout"))
        fast_sparse = await _make_fast_retriever("sparse_only")

        with patch.object(
            search_service, "_bm25_recall", side_effect=slow
        ), patch.object(
            search_service, "_dense_recall", side_effect=failing
        ), patch.object(
            search_service, "_sparse_recall", side_effect=fast_sparse
        ):
            results = await search_service._multi_recall(
                query="q",
                dense_vector=[0.1] * 1024,
                sparse_indices=[1],
                sparse_values=[0.5],
                qdrant_filter={},
                opensearch_filter={},
            )

        assert len(results) == 1
        assert results[0][0].chunk_id == "sparse_only"


# ─── Concurrent success (no timeout fires) ────────────────────────────


class TestConcurrentSuccess:
    """三路全部成功且并发执行：返回三个非空列表，总耗时贴近最慢路。"""

    @pytest.mark.asyncio
    async def test_all_three_paths_succeed(
        self, search_service: SearchService, short_timeout: float
    ) -> None:
        """三路在超时窗口内均返回 → ``_multi_recall`` 返回 3 个列表。"""
        fast_bm25 = await _make_fast_retriever("bm25")
        fast_dense = await _make_fast_retriever("dense")
        fast_sparse = await _make_fast_retriever("sparse")

        with patch.object(
            search_service, "_bm25_recall", side_effect=fast_bm25
        ), patch.object(
            search_service, "_dense_recall", side_effect=fast_dense
        ), patch.object(
            search_service, "_sparse_recall", side_effect=fast_sparse
        ):
            results = await search_service._multi_recall(
                query="q",
                dense_vector=[0.1] * 1024,
                sparse_indices=[1],
                sparse_values=[0.5],
                qdrant_filter={},
                opensearch_filter={},
            )

        assert len(results) == 3
        chunk_ids = {hit.chunk_id for path in results for hit in path}
        assert chunk_ids == {"bm25", "dense", "sparse"}

    @pytest.mark.asyncio
    async def test_paths_run_concurrently_not_serially(
        self, search_service: SearchService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """三路应并发执行：总耗时应贴近最慢一路，远低于三路串行总耗时。"""
        # 关闭超时影响：把超时设大一些，确保所有路都能完成
        monkeypatch.setattr(search_service_module, "RETRIEVER_TIMEOUT", 5.0)

        # 每路睡 0.1 秒；并发应在 ~0.1s 完成，串行需 ~0.3s
        slow_bm25 = await _make_slow_retriever(delay=0.1, chunk_id="bm25")
        slow_dense = await _make_slow_retriever(delay=0.1, chunk_id="dense")
        slow_sparse = await _make_slow_retriever(delay=0.1, chunk_id="sparse")

        with patch.object(
            search_service, "_bm25_recall", side_effect=slow_bm25
        ), patch.object(
            search_service, "_dense_recall", side_effect=slow_dense
        ), patch.object(
            search_service, "_sparse_recall", side_effect=slow_sparse
        ):
            t0 = asyncio.get_event_loop().time()
            results = await search_service._multi_recall(
                query="q",
                dense_vector=[0.1] * 1024,
                sparse_indices=[1],
                sparse_values=[0.5],
                qdrant_filter={},
                opensearch_filter={},
            )
            elapsed = asyncio.get_event_loop().time() - t0

        assert len(results) == 3
        # 并发执行：总耗时应明显小于 0.3s（三路串行总和），留出一定调度余量
        assert elapsed < 0.25, (
            f"Multi-recall should run concurrently; elapsed {elapsed:.3f}s "
            f"suggests serial execution"
        )

    @pytest.mark.asyncio
    async def test_slow_path_does_not_block_fast_paths_until_timeout(
        self, search_service: SearchService, short_timeout: float
    ) -> None:
        """快路应在自身完成时立刻返回，不会被慢路阻塞超过 timeout 窗口。

        慢路超时窗口为 ``short_timeout``（0.1s），快路 ~0；总耗时应贴近
        ``short_timeout`` 而非更长。
        """
        slow = await _make_slow_retriever(delay=short_timeout * 10)
        fast = await _make_fast_retriever("fast")

        with patch.object(
            search_service, "_bm25_recall", side_effect=slow
        ), patch.object(
            search_service, "_dense_recall", side_effect=fast
        ), patch.object(
            search_service, "_sparse_recall", side_effect=fast
        ):
            t0 = asyncio.get_event_loop().time()
            results = await search_service._multi_recall(
                query="q",
                dense_vector=[0.1] * 1024,
                sparse_indices=[1],
                sparse_values=[0.5],
                qdrant_filter={},
                opensearch_filter={},
            )
            elapsed = asyncio.get_event_loop().time() - t0

        # 慢路按超时时间 short_timeout 被取消，所以总耗时应贴近 short_timeout
        assert len(results) == 2
        assert elapsed < short_timeout * 3, (
            f"Total elapsed {elapsed:.3f}s exceeds reasonable bound "
            f"({short_timeout * 3:.3f}s) — fast paths may be blocked"
        )
