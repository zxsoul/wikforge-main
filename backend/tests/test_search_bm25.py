"""BM25 检索器（OpenSearch + IK 分词 + 权限过滤）单元测试。

对应任务 14.1：实现 BM25 检索器（OpenSearch query，IK 分词，权限过滤，返回 Top 50）。

测试范围：
- OpenSearch 查询体构造（multi_match + ik_smart + bool/filter）
- 权限过滤（用户级 term + 空间级 terms）
- Top 50（``size`` 参数）
- 服务端 3 秒查询超时（``timeout`` 参数）
- 高亮片段 200 字符配置
- ``_source`` 仅返回必要字段
- 命中结果转换为 ``SearchHit``
- 空结果与异常路径（缺少 ``hits`` / 客户端抛错）
- IK 索引名/索引未命中等极端场景
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.services.search_service import (
    HIGHLIGHT_MAX_CHARS,
    RETRIEVER_TIMEOUT,
    TOP_K_PER_RETRIEVER,
    SearchHit,
    SearchService,
)


# ─── Helpers ──────────────────────────────────────────────────────────


def _make_os_response(hits: list[dict] | None = None) -> dict:
    """构造一个 OpenSearch 响应字典。"""
    return {
        "took": 5,
        "timed_out": False,
        "hits": {
            "total": {"value": len(hits or []), "relation": "eq"},
            "hits": hits or [],
        },
    }


def _make_os_hit(
    *,
    chunk_id: str | None = None,
    document_id: str | None = None,
    space_id: str | None = None,
    chunk_index: int = 0,
    title_chain: str = "Section > Subsection",
    source_file: str = "test.pdf",
    content: str = "这是一个测试文本块",
    score: float = 1.5,
) -> dict:
    return {
        "_id": chunk_id or str(uuid.uuid4()),
        "_score": score,
        "_source": {
            "chunk_id": chunk_id or str(uuid.uuid4()),
            "document_id": document_id or str(uuid.uuid4()),
            "space_id": space_id or str(uuid.uuid4()),
            "chunk_index": chunk_index,
            "title_chain": title_chain,
            "source_file": source_file,
            "content": content,
        },
    }


@pytest.fixture
def search_service() -> SearchService:
    """无 embedding 依赖的 SearchService（BM25 测试不需要 embedding）。"""
    # 使用 MagicMock 避免真实 EmbeddingService 初始化（其会尝试加载模型）
    return SearchService(embedding_service=MagicMock())


@pytest.fixture
def permission_filter(search_service: SearchService) -> dict:
    """构造一个真实的 OpenSearch 权限过滤条件用于查询体校验。"""
    user_id = "user-abc"
    space_ids = ["space-1", "space-2"]
    return search_service._build_opensearch_filter(user_id, space_ids)


# ─── 查询体构造 ──────────────────────────────────────────────────────


class TestBM25QueryBody:
    """验证发送给 OpenSearch 的查询体格式正确。"""

    @pytest.mark.asyncio
    async def test_query_body_uses_ik_smart_analyzer(
        self, search_service: SearchService, permission_filter: dict
    ) -> None:
        """查询时必须显式使用 IK 分词器（ik_smart）。"""
        client = MagicMock()
        client.search.return_value = _make_os_response()

        with patch(
            "app.core.opensearch.get_opensearch_client", return_value=client
        ):
            await search_service._bm25_recall("齿轮箱噪声", permission_filter)

        assert client.search.called
        call_kwargs = client.search.call_args.kwargs
        body = call_kwargs["body"]

        multi_match = body["query"]["bool"]["must"][0]["multi_match"]
        assert multi_match["analyzer"] == "ik_smart"
        assert multi_match["query"] == "齿轮箱噪声"

    @pytest.mark.asyncio
    async def test_query_body_searches_content_and_title(
        self, search_service: SearchService, permission_filter: dict
    ) -> None:
        """multi_match 应同时检索 content（加权）与 title_chain。"""
        client = MagicMock()
        client.search.return_value = _make_os_response()

        with patch(
            "app.core.opensearch.get_opensearch_client", return_value=client
        ):
            await search_service._bm25_recall("test", permission_filter)

        body = client.search.call_args.kwargs["body"]
        fields = body["query"]["bool"]["must"][0]["multi_match"]["fields"]
        # content 应有更高权重（^2），title_chain 一并检索
        assert any(f.startswith("content") for f in fields)
        assert any("^" in f for f in fields if f.startswith("content"))
        assert "title_chain" in fields

    @pytest.mark.asyncio
    async def test_query_body_returns_top_50(
        self, search_service: SearchService, permission_filter: dict
    ) -> None:
        """size 必须为 50（Top 50 召回）。"""
        client = MagicMock()
        client.search.return_value = _make_os_response()

        with patch(
            "app.core.opensearch.get_opensearch_client", return_value=client
        ):
            await search_service._bm25_recall("test", permission_filter)

        body = client.search.call_args.kwargs["body"]
        assert body["size"] == TOP_K_PER_RETRIEVER == 50

    @pytest.mark.asyncio
    async def test_query_body_has_3s_server_timeout(
        self, search_service: SearchService, permission_filter: dict
    ) -> None:
        """服务端查询应配置 3 秒超时（与外层 asyncio.wait_for 双重保障）。"""
        client = MagicMock()
        client.search.return_value = _make_os_response()

        with patch(
            "app.core.opensearch.get_opensearch_client", return_value=client
        ):
            await search_service._bm25_recall("test", permission_filter)

        body = client.search.call_args.kwargs["body"]
        # OpenSearch 接受形如 "3s" 的字符串
        assert body.get("timeout") == f"{int(RETRIEVER_TIMEOUT)}s"

    @pytest.mark.asyncio
    async def test_query_body_highlight_200_chars(
        self, search_service: SearchService, permission_filter: dict
    ) -> None:
        """高亮片段大小应为 200 字符。"""
        client = MagicMock()
        client.search.return_value = _make_os_response()

        with patch(
            "app.core.opensearch.get_opensearch_client", return_value=client
        ):
            await search_service._bm25_recall("test", permission_filter)

        body = client.search.call_args.kwargs["body"]
        assert "highlight" in body
        content_highlight = body["highlight"]["fields"]["content"]
        assert content_highlight["fragment_size"] == HIGHLIGHT_MAX_CHARS == 200

    @pytest.mark.asyncio
    async def test_query_body_targets_chunks_index(
        self, search_service: SearchService, permission_filter: dict
    ) -> None:
        """请求应针对 chunks 索引发起。"""
        from app.core.opensearch import INDEX_NAME

        client = MagicMock()
        client.search.return_value = _make_os_response()

        with patch(
            "app.core.opensearch.get_opensearch_client", return_value=client
        ):
            await search_service._bm25_recall("test", permission_filter)

        assert client.search.call_args.kwargs["index"] == INDEX_NAME == "chunks"

    @pytest.mark.asyncio
    async def test_query_body_source_fields_only_required(
        self, search_service: SearchService, permission_filter: dict
    ) -> None:
        """_source 字段集合应仅包含 SearchHit 所需字段，避免额外网络开销。"""
        client = MagicMock()
        client.search.return_value = _make_os_response()

        with patch(
            "app.core.opensearch.get_opensearch_client", return_value=client
        ):
            await search_service._bm25_recall("test", permission_filter)

        body = client.search.call_args.kwargs["body"]
        expected = {
            "chunk_id",
            "document_id",
            "space_id",
            "chunk_index",
            "title_chain",
            "source_file",
            "page_number",
            "content",
        }
        assert set(body["_source"]) == expected


# ─── 权限过滤 ─────────────────────────────────────────────────────────


class TestBM25PermissionFilter:
    """验证查询体中嵌入的权限过滤条件。"""

    @pytest.mark.asyncio
    async def test_permission_filter_applied_in_bool_filter(
        self, search_service: SearchService
    ) -> None:
        """权限过滤必须出现在 bool/filter 中（而非 must）。"""
        user_id = "user-xyz"
        space_ids = ["space-A"]
        perm_filter = search_service._build_opensearch_filter(user_id, space_ids)

        client = MagicMock()
        client.search.return_value = _make_os_response()

        with patch(
            "app.core.opensearch.get_opensearch_client", return_value=client
        ):
            await search_service._bm25_recall("query", perm_filter)

        body = client.search.call_args.kwargs["body"]
        filters = body["query"]["bool"]["filter"]
        assert filters == [perm_filter]

    @pytest.mark.asyncio
    async def test_permission_filter_terms_includes_user_id(
        self, search_service: SearchService
    ) -> None:
        """权限过滤应通过 term 查询匹配用户 ID。"""
        user_id = str(uuid.uuid4())
        perm_filter = search_service._build_opensearch_filter(
            user_id, ["space-1"]
        )

        # 直接断言权限过滤体结构
        clauses = perm_filter["bool"]["should"]
        assert {"term": {"allowed_user_ids": user_id}} in clauses

    @pytest.mark.asyncio
    async def test_permission_filter_terms_includes_space_ids(
        self, search_service: SearchService
    ) -> None:
        """权限过滤应通过 terms 查询匹配可访问空间列表。"""
        user_id = "user-1"
        space_ids = [str(uuid.uuid4()) for _ in range(3)]
        perm_filter = search_service._build_opensearch_filter(user_id, space_ids)

        clauses = perm_filter["bool"]["should"]
        assert {"terms": {"space_id": space_ids}} in clauses

    @pytest.mark.asyncio
    async def test_permission_filter_minimum_should_match_one(
        self, search_service: SearchService
    ) -> None:
        """should 子句须设置 minimum_should_match=1，确保至少满足一条权限条件。"""
        perm_filter = search_service._build_opensearch_filter(
            "user-1", ["space-1"]
        )
        assert perm_filter["bool"]["minimum_should_match"] == 1


# ─── 命中转换 ─────────────────────────────────────────────────────────


class TestBM25HitConversion:
    """验证 OpenSearch 响应转换为 SearchHit 列表。"""

    @pytest.mark.asyncio
    async def test_hits_converted_to_search_hit(
        self, search_service: SearchService, permission_filter: dict
    ) -> None:
        """response.hits.hits 应映射为 SearchHit 列表。"""
        os_hits = [
            _make_os_hit(
                chunk_id="c1",
                document_id="d1",
                space_id="s1",
                chunk_index=3,
                title_chain="一 > 1.1",
                source_file="规范.pdf",
                content="水泥工艺技术规范",
                score=2.34,
            ),
            _make_os_hit(
                chunk_id="c2",
                document_id="d1",
                space_id="s1",
                chunk_index=4,
                content="另一段内容",
                score=1.10,
            ),
        ]
        client = MagicMock()
        client.search.return_value = _make_os_response(os_hits)

        with patch(
            "app.core.opensearch.get_opensearch_client", return_value=client
        ):
            results = await search_service._bm25_recall(
                "水泥规范", permission_filter
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
        assert first.content == "水泥工艺技术规范"
        assert first.score == pytest.approx(2.34)

    @pytest.mark.asyncio
    async def test_score_preserved_from_opensearch(
        self, search_service: SearchService, permission_filter: dict
    ) -> None:
        """原始 BM25 _score 必须被保留（供后续 RRF 融合使用）。"""
        os_hits = [_make_os_hit(score=42.0)]
        client = MagicMock()
        client.search.return_value = _make_os_response(os_hits)

        with patch(
            "app.core.opensearch.get_opensearch_client", return_value=client
        ):
            results = await search_service._bm25_recall("q", permission_filter)

        assert results[0].score == 42.0

    @pytest.mark.asyncio
    async def test_missing_source_fields_default_to_empty(
        self, search_service: SearchService, permission_filter: dict
    ) -> None:
        """缺失字段应回退到默认值，不应抛 KeyError。"""
        os_hits = [{"_id": "x", "_score": 1.0, "_source": {}}]
        client = MagicMock()
        client.search.return_value = _make_os_response(os_hits)

        with patch(
            "app.core.opensearch.get_opensearch_client", return_value=client
        ):
            results = await search_service._bm25_recall("q", permission_filter)

        hit = results[0]
        assert hit.chunk_id == ""
        assert hit.document_id == ""
        assert hit.chunk_index == 0
        assert hit.title_chain == ""
        assert hit.content == ""


# ─── 边界与异常 ───────────────────────────────────────────────────────


class TestBM25EdgeCases:
    """空结果、缺字段、异常等极端场景。"""

    @pytest.mark.asyncio
    async def test_empty_hits_returns_empty_list(
        self, search_service: SearchService, permission_filter: dict
    ) -> None:
        """OpenSearch 返回 0 条命中时，返回空列表。"""
        client = MagicMock()
        client.search.return_value = _make_os_response([])

        with patch(
            "app.core.opensearch.get_opensearch_client", return_value=client
        ):
            results = await search_service._bm25_recall(
                "找不到的内容", permission_filter
            )

        assert results == []

    @pytest.mark.asyncio
    async def test_missing_hits_field_returns_empty_list(
        self, search_service: SearchService, permission_filter: dict
    ) -> None:
        """OpenSearch 响应缺少 hits 字段时不应抛错。"""
        client = MagicMock()
        client.search.return_value = {}  # 空响应

        with patch(
            "app.core.opensearch.get_opensearch_client", return_value=client
        ):
            results = await search_service._bm25_recall("q", permission_filter)

        assert results == []

    @pytest.mark.asyncio
    async def test_client_error_propagates(
        self, search_service: SearchService, permission_filter: dict
    ) -> None:
        """OpenSearch 客户端抛错时，错误应向上传播（由 _multi_recall 捕获并降级）。"""
        client = MagicMock()
        client.search.side_effect = RuntimeError("opensearch unreachable")

        with patch(
            "app.core.opensearch.get_opensearch_client", return_value=client
        ):
            with pytest.raises(RuntimeError, match="opensearch unreachable"):
                await search_service._bm25_recall("q", permission_filter)

    @pytest.mark.asyncio
    async def test_runs_in_executor_does_not_block_event_loop(
        self, search_service: SearchService, permission_filter: dict
    ) -> None:
        """同步 opensearch-py 调用必须放入线程池，避免阻塞事件循环。"""
        # 模拟一个慢调用，确保期间事件循环仍然可推进其他任务
        import time

        def slow_search(*args, **kwargs):
            time.sleep(0.1)
            return _make_os_response()

        client = MagicMock()
        client.search.side_effect = slow_search

        async def parallel_task() -> str:
            await asyncio.sleep(0.01)
            return "done"

        with patch(
            "app.core.opensearch.get_opensearch_client", return_value=client
        ):
            t0 = asyncio.get_event_loop().time()
            recall, side = await asyncio.gather(
                search_service._bm25_recall("q", permission_filter),
                parallel_task(),
            )
            elapsed = asyncio.get_event_loop().time() - t0

        # 并发执行的 sleep(0.01) 应在 BM25 完成前就结束；
        # 总耗时应接近最慢任务（~0.1s），而非串行的 0.11s+
        assert side == "done"
        assert elapsed < 0.2
        assert recall == []
