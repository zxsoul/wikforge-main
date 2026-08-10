"""复合搜索引擎端到端集成测试（任务 14.10）。

与已有的 ``test_search_*.py`` 单元测试互补：单元测试在 SearchService 方法
层面打桩，验证每一段算法的契约；本文件则把 API 路由 → ``SearchService``
→ OpenSearch / Qdrant 客户端三层串联起来，仅在最外侧的客户端层 mock
返回值，从而真正执行多路召回、RRF 融合、Cross-Encoder 精排（降级到
关键词重叠）、结果格式化与分页的完整链路。

覆盖点（对应需求 6 整体验收标准）：
- 端到端流程：query → 多路召回 → RRF → rerank → 格式化 → 分页
- 权限隔离：用户 A 不能搜到用户 B 的私有空间内容
- 边界场景：
    * 空白查询（绕过 ``min_length=1`` 但不应崩溃）
    * 超长查询（500 字以内的最大值）
    * 特殊字符（引号、CJK、emoji、转义符）
    * 单一路径返回（其他路超时被降级跳过）
    * 三路皆空 → 返回空结果 200
- 全局 5 秒超时与单路 3 秒超时的协作

设计参考：
- ``app/services/search_service.py``：SearchService 主入口
- ``app/api/search.py``：POST /api/search 路由
- ``Requirements 6.1–6.7``
"""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.services.search_service as search_service_module
from app.api.auth import get_current_user
from app.api.search import get_search_service
from app.api.search import router as search_router
from app.core.database import get_db
from app.core.exceptions import register_exception_handlers
from app.services.search_service import (
    HIGHLIGHT_MARK_OPEN,
    HIGHLIGHT_MAX_CHARS,
    SearchService,
)

# ─── Helpers ──────────────────────────────────────────────────────────


def _make_os_hit(
    *,
    chunk_id: str,
    document_id: str,
    space_id: str,
    chunk_index: int = 0,
    title_chain: str = "Section > Subsection",
    source_file: str = "demo.pdf",
    page_number: int = 1,
    content: str = "示例正文",
    score: float = 1.5,
    allowed_user_ids: list[str] | None = None,
) -> dict:
    """构造一条 OpenSearch ``hits.hits[i]`` 数据（``_source`` 形态）。"""
    return {
        "_id": chunk_id,
        "_score": score,
        "_source": {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "space_id": space_id,
            "chunk_index": chunk_index,
            "title_chain": title_chain,
            "source_file": source_file,
            "page_number": page_number,
            "content": content,
            "allowed_user_ids": allowed_user_ids or [],
        },
    }


def _make_qdrant_point(
    *,
    chunk_id: str,
    document_id: str,
    space_id: str,
    chunk_index: int = 0,
    title_chain: str = "Section > Subsection",
    source_file: str = "demo.pdf",
    page_number: int = 1,
    content: str = "示例正文",
    score: float = 0.85,
    allowed_user_ids: list[str] | None = None,
) -> SimpleNamespace:
    """构造一个 Qdrant ScoredPoint 等价对象。"""
    return SimpleNamespace(
        id=chunk_id,
        score=score,
        payload={
            "document_id": document_id,
            "space_id": space_id,
            "chunk_index": chunk_index,
            "title_chain": title_chain,
            "source_file": source_file,
            "page_number": page_number,
            "content": content,
            "allowed_user_ids": allowed_user_ids or [],
        },
    )


def _opensearch_response(hits: list[dict]) -> dict:
    """组装 OpenSearch 响应 envelope。"""
    return {
        "took": 5,
        "timed_out": False,
        "hits": {
            "total": {"value": len(hits), "relation": "eq"},
            "hits": hits,
        },
    }


def _filter_os_hits_by_permission(
    all_hits: list[dict], user_id: str, space_ids: list[str]
) -> list[dict]:
    """模拟 OpenSearch 服务端按权限过滤。

    OR 语义：``allowed_user_ids`` 命中当前用户 **或** ``space_id`` 属于
    可访问空间集合。这里用 Python 复刻服务端行为，让测试中的 mock 足以
    展示真实 Pre-Filtering 效果。
    """
    space_set = set(space_ids)
    out: list[dict] = []
    for hit in all_hits:
        src = hit["_source"]
        allowed_users = set(src.get("allowed_user_ids", []) or [])
        if user_id in allowed_users or src.get("space_id") in space_set:
            out.append(hit)
    return out


def _filter_points_by_permission(
    all_points: list[SimpleNamespace], user_id: str, space_ids: list[str]
) -> list[SimpleNamespace]:
    """模拟 Qdrant 服务端按权限过滤（与 OpenSearch 一致的 OR 语义）。"""
    space_set = set(space_ids)
    out: list[SimpleNamespace] = []
    for p in all_points:
        payload = p.payload or {}
        allowed_users = set(payload.get("allowed_user_ids", []) or [])
        if user_id in allowed_users or payload.get("space_id") in space_set:
            out.append(p)
    return out


def _build_app(
    *,
    user_id: uuid.UUID,
    allowed_space_ids: list[str],
    search_service: SearchService,
) -> FastAPI:
    """构造一个隔离的 FastAPI app，用 mock 的依赖替换鉴权 + DB + Service。"""
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(search_router)

    fake_user = MagicMock()
    fake_user.id = user_id

    async def _override_user():
        return fake_user

    db_session = AsyncMock()
    db_result = MagicMock()
    db_result.scalars.return_value.all.return_value = list(allowed_space_ids)
    db_session.execute = AsyncMock(return_value=db_result)

    async def _override_db():
        yield db_session

    app.dependency_overrides[get_current_user] = _override_user
    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_search_service] = lambda: search_service
    return app


def _build_embedding_mock() -> AsyncMock:
    """构造一个返回稳定向量的 EmbeddingService mock。

    这里只关心向量被透传到 Qdrant，不关心其内容是否与真实嵌入语义一致。
    """
    service = AsyncMock()
    service.embed_query = AsyncMock(
        return_value=MagicMock(
            chunk_id="query",
            dense_vector=[0.01] * 1024,
            sparse_indices=[1, 7, 42, 137],
            sparse_values=[0.5, 0.3, 0.2, 0.1],
        )
    )
    return service


# ─── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def user_a() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def user_b() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def space_a_public() -> str:
    """用户 A 与 B 共享的公共空间。"""
    return str(uuid.uuid4())


@pytest.fixture
def space_a_private() -> str:
    """仅用户 A 可访问的私有空间。"""
    return str(uuid.uuid4())


@pytest.fixture
def space_b_private() -> str:
    """仅用户 B 可访问的私有空间。"""
    return str(uuid.uuid4())


@pytest.fixture
def search_service_with_mock_embedding() -> SearchService:
    """注入 mock embedding 的真实 SearchService 实例。

    SearchService 内部的 ``_bm25_recall`` / ``_dense_recall`` /
    ``_sparse_recall`` 等方法都是真实代码，仅 ``EmbeddingService`` 与
    ``OpenSearch``/``Qdrant`` 客户端被 mock。
    """
    return SearchService(embedding_service=_build_embedding_mock())


# ─── 端到端流程 ───────────────────────────────────────────────────────


class TestEndToEndSearchFlow:
    """query → 多路召回 → RRF → rerank → 格式化 → 分页。"""

    def test_full_flow_returns_paginated_ranked_results(
        self,
        search_service_with_mock_embedding: SearchService,
        user_a: uuid.UUID,
        space_a_public: str,
    ) -> None:
        """三路召回均返回内容时，API 输出经过 RRF + 精排的分页结果。"""
        # 三个 chunk：
        # - shared 同时出现在 BM25 / Dense / Sparse → RRF 后排第一
        # - bm25_only 仅 BM25 命中
        # - dense_only 仅 Dense 命中
        os_hits = [
            _make_os_hit(
                chunk_id="shared",
                document_id="doc-1",
                space_id=space_a_public,
                content="复合搜索引擎使用 BM25 与向量召回",
                score=8.0,
            ),
            _make_os_hit(
                chunk_id="bm25_only",
                document_id="doc-2",
                space_id=space_a_public,
                content="BM25 关键词命中段落",
                score=5.0,
            ),
        ]
        dense_points = [
            _make_qdrant_point(
                chunk_id="shared",
                document_id="doc-1",
                space_id=space_a_public,
                content="复合搜索引擎使用 BM25 与向量召回",
                score=0.92,
            ),
            _make_qdrant_point(
                chunk_id="dense_only",
                document_id="doc-3",
                space_id=space_a_public,
                content="语义向量命中段落",
                score=0.71,
            ),
        ]
        sparse_points = [
            _make_qdrant_point(
                chunk_id="shared",
                document_id="doc-1",
                space_id=space_a_public,
                content="复合搜索引擎使用 BM25 与向量召回",
                score=0.65,
            ),
        ]

        os_client = MagicMock()
        os_client.search.return_value = _opensearch_response(os_hits)

        # Qdrant 客户端区分 dense / sparse 调用（依据 query_vector 类型）
        qdrant_client = MagicMock()

        def _qdrant_search(**kwargs):
            from qdrant_client.models import NamedSparseVector, NamedVector

            qv = kwargs["query_vector"]
            if isinstance(qv, NamedVector) and qv.name == "dense":
                return dense_points
            if isinstance(qv, NamedSparseVector) and qv.name == "sparse":
                return sparse_points
            return []

        qdrant_client.search.side_effect = _qdrant_search

        with patch(
            "app.core.opensearch.get_opensearch_client", return_value=os_client
        ), patch(
            "app.core.qdrant.get_qdrant_client", return_value=qdrant_client
        ):
            app = _build_app(
                user_id=user_a,
                allowed_space_ids=[space_a_public],
                search_service=search_service_with_mock_embedding,
            )
            client = TestClient(app)
            resp = client.post(
                "/api/search",
                json={"query": "复合搜索 BM25 向量", "page": 1, "page_size": 10},
            )

        assert resp.status_code == 200, resp.text
        body = resp.json()

        # 1) 总数应为去重后的候选数（3 条），page=1, page_size=10
        assert body["total"] == 3
        assert body["page"] == 1
        assert body["page_size"] == 10
        assert len(body["results"]) == 3

        # 2) 多路命中的 shared 应排在最前
        assert body["results"][0]["chunk_id"] == "shared"

        # 3) 每条结果的 score ∈ [0, 1]，highlight ≤ 200 字符
        for item in body["results"]:
            assert 0.0 <= item["score"] <= 1.0
            assert len(item["highlight"]) <= HIGHLIGHT_MAX_CHARS

        # 4) 来源信息字段齐全
        first = body["results"][0]
        assert first["document_id"] == "doc-1"
        assert first["title_chain"] == "Section > Subsection"
        assert first["source_file"] == "demo.pdf"
        assert "page_number" in first

        # 5) 三路客户端都应该被调用一次（OpenSearch 1 次 + Qdrant 2 次）
        assert os_client.search.call_count == 1
        assert qdrant_client.search.call_count == 2

    def test_pagination_returns_correct_slice(
        self,
        search_service_with_mock_embedding: SearchService,
        user_a: uuid.UUID,
        space_a_public: str,
    ) -> None:
        """page=2, page_size=5 应返回 RRF 排序后第 6-10 条。"""
        # 30 条互不重叠的 BM25 命中
        os_hits = [
            _make_os_hit(
                chunk_id=f"c{i:02d}",
                document_id=f"doc-{i}",
                space_id=space_a_public,
                content=f"搜索结果 {i}",
                score=10.0 - i * 0.1,
            )
            for i in range(30)
        ]
        os_client = MagicMock()
        os_client.search.return_value = _opensearch_response(os_hits)

        qdrant_client = MagicMock()
        qdrant_client.search.return_value = []

        with patch(
            "app.core.opensearch.get_opensearch_client", return_value=os_client
        ), patch(
            "app.core.qdrant.get_qdrant_client", return_value=qdrant_client
        ):
            app = _build_app(
                user_id=user_a,
                allowed_space_ids=[space_a_public],
                search_service=search_service_with_mock_embedding,
            )
            client = TestClient(app)
            resp = client.post(
                "/api/search",
                json={"query": "搜索", "page": 2, "page_size": 5},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 30
        assert body["page"] == 2
        assert body["page_size"] == 5
        assert len(body["results"]) == 5

        # 第 2 页应是排序后的第 6-10 条（c05..c09）
        chunk_ids = [r["chunk_id"] for r in body["results"]]
        assert chunk_ids == [f"c{i:02d}" for i in range(5, 10)]

    def test_empty_recalls_returns_empty_results(
        self,
        search_service_with_mock_embedding: SearchService,
        user_a: uuid.UUID,
        space_a_public: str,
    ) -> None:
        """三路召回都为空 → API 返回 200 且 results 为空。"""
        os_client = MagicMock()
        os_client.search.return_value = _opensearch_response([])
        qdrant_client = MagicMock()
        qdrant_client.search.return_value = []

        with patch(
            "app.core.opensearch.get_opensearch_client", return_value=os_client
        ), patch(
            "app.core.qdrant.get_qdrant_client", return_value=qdrant_client
        ):
            app = _build_app(
                user_id=user_a,
                allowed_space_ids=[space_a_public],
                search_service=search_service_with_mock_embedding,
            )
            client = TestClient(app)
            resp = client.post(
                "/api/search", json={"query": "找不到的内容"}
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["results"] == []


# ─── 权限隔离 ─────────────────────────────────────────────────────────


class TestPermissionIsolation:
    """用户 A 不能搜到仅 B 可见的私有空间内容。"""

    def test_user_a_cannot_see_user_b_private_space_content(
        self,
        search_service_with_mock_embedding: SearchService,
        user_a: uuid.UUID,
        user_b: uuid.UUID,
        space_a_public: str,
        space_b_private: str,
    ) -> None:
        """OpenSearch / Qdrant 端 Pre-Filtering 应剔除 B 私有空间的命中。"""
        # 全量数据：3 条 chunk
        # - public: 公共空间，A 可见
        # - b_private: B 的私有空间，A 不可见
        # - b_explicit_share: 仅通过 allowed_user_ids 显式分享给用户 B
        all_os_hits = [
            _make_os_hit(
                chunk_id="public",
                document_id="doc-pub",
                space_id=space_a_public,
                content="公共内容人人可见",
                score=5.0,
            ),
            _make_os_hit(
                chunk_id="b_private",
                document_id="doc-bp",
                space_id=space_b_private,
                content="B 的私密笔记",
                score=4.5,
                allowed_user_ids=[str(user_b)],
            ),
            _make_os_hit(
                chunk_id="b_explicit_share",
                document_id="doc-bs",
                space_id=space_b_private,
                content="B 单独分享给某人",
                score=4.0,
                allowed_user_ids=[str(user_b)],
            ),
        ]
        all_dense_points = [
            _make_qdrant_point(
                chunk_id="public",
                document_id="doc-pub",
                space_id=space_a_public,
                content="公共内容人人可见",
                score=0.9,
            ),
            _make_qdrant_point(
                chunk_id="b_private",
                document_id="doc-bp",
                space_id=space_b_private,
                content="B 的私密笔记",
                score=0.85,
                allowed_user_ids=[str(user_b)],
            ),
        ]

        os_client = MagicMock()
        qdrant_client = MagicMock()

        # OpenSearch 端按 body 中的权限 filter 过滤
        def _os_search(*, index, body, **_):
            # 拆解出 user_id 与 space_ids
            should = body["query"]["bool"]["filter"][0]["bool"]["should"]
            uid = next(
                c["term"]["allowed_user_ids"]
                for c in should
                if "term" in c
            )
            sids = next(
                c["terms"]["space_id"] for c in should if "terms" in c
            )
            return _opensearch_response(
                _filter_os_hits_by_permission(all_os_hits, uid, sids)
            )

        os_client.search.side_effect = _os_search

        # Qdrant 端按 query_filter 内的 should 子句过滤
        def _qdrant_search(**kwargs):
            from qdrant_client.models import (
                FieldCondition,
                MatchAny,
                MatchValue,
                NamedSparseVector,
                NamedVector,
            )

            qf = kwargs["query_filter"]
            uid = ""
            sids: list[str] = []
            for cond in qf.should or []:
                if isinstance(cond, FieldCondition):
                    if cond.key == "allowed_user_ids" and isinstance(
                        cond.match, MatchValue
                    ):
                        uid = cond.match.value
                    elif cond.key == "space_id" and isinstance(
                        cond.match, MatchAny
                    ):
                        sids = list(cond.match.any)

            qv = kwargs["query_vector"]
            if isinstance(qv, NamedVector) and qv.name == "dense":
                return _filter_points_by_permission(all_dense_points, uid, sids)
            if isinstance(qv, NamedSparseVector) and qv.name == "sparse":
                return []
            return []

        qdrant_client.search.side_effect = _qdrant_search

        with patch(
            "app.core.opensearch.get_opensearch_client", return_value=os_client
        ), patch(
            "app.core.qdrant.get_qdrant_client", return_value=qdrant_client
        ):
            app = _build_app(
                user_id=user_a,
                allowed_space_ids=[space_a_public],  # A 不在 B 的私有空间
                search_service=search_service_with_mock_embedding,
            )
            client = TestClient(app)
            resp = client.post(
                "/api/search", json={"query": "笔记 内容"}
            )

        assert resp.status_code == 200
        body = resp.json()

        chunk_ids = {r["chunk_id"] for r in body["results"]}
        assert "public" in chunk_ids, "公共内容应可被 A 检索到"
        assert "b_private" not in chunk_ids, "B 私有空间内容不能泄漏给 A"
        assert "b_explicit_share" not in chunk_ids, (
            "未显式分享给 A 的内容不能出现在 A 的搜索结果"
        )

    def test_user_with_no_spaces_gets_empty_results(
        self,
        search_service_with_mock_embedding: SearchService,
        user_a: uuid.UUID,
    ) -> None:
        """无任何可访问空间的用户应直接拿到空结果，且不会发起后端调用。"""
        os_client = MagicMock()
        qdrant_client = MagicMock()

        with patch(
            "app.core.opensearch.get_opensearch_client", return_value=os_client
        ), patch(
            "app.core.qdrant.get_qdrant_client", return_value=qdrant_client
        ):
            app = _build_app(
                user_id=user_a,
                allowed_space_ids=[],  # 没有任何空间权限
                search_service=search_service_with_mock_embedding,
            )
            client = TestClient(app)
            resp = client.post(
                "/api/search", json={"query": "任意查询"}
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["results"] == []
        # SearchService 在 allowed_space_ids 为空时短路，不应触达检索后端
        assert os_client.search.call_count == 0
        assert qdrant_client.search.call_count == 0


# ─── 单一路径返回（其他路超时） ────────────────────────────────────────


class TestSinglePathRecallSurvives:
    """两路超时时，仅剩的一路结果仍能完成 RRF + 精排 + 分页。"""

    def test_only_bm25_returns_others_time_out(
        self,
        search_service_with_mock_embedding: SearchService,
        user_a: uuid.UUID,
        space_a_public: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Dense 与 Sparse 双双超时，BM25 单路结果仍能成功返回。

        通过把 ``RETRIEVER_TIMEOUT`` 缩到 0.05 秒并让 Qdrant 客户端 sleep
        来触发 ``asyncio.wait_for`` 的超时分支，避免测试等待 3 秒。
        """
        monkeypatch.setattr(search_service_module, "RETRIEVER_TIMEOUT", 0.05)

        os_hits = [
            _make_os_hit(
                chunk_id=f"bm25_{i}",
                document_id=f"doc-{i}",
                space_id=space_a_public,
                content=f"BM25 仅这一路返回 {i}",
                score=10.0 - i,
            )
            for i in range(3)
        ]

        os_client = MagicMock()
        os_client.search.return_value = _opensearch_response(os_hits)

        # Qdrant 同步调用会被 run_in_executor 包装；
        # 让其 sleep 远超 RETRIEVER_TIMEOUT，触发外层超时
        import time as _time

        def _slow_qdrant(**_kwargs):
            _time.sleep(0.5)
            return []

        qdrant_client = MagicMock()
        qdrant_client.search.side_effect = _slow_qdrant

        with patch(
            "app.core.opensearch.get_opensearch_client", return_value=os_client
        ), patch(
            "app.core.qdrant.get_qdrant_client", return_value=qdrant_client
        ):
            app = _build_app(
                user_id=user_a,
                allowed_space_ids=[space_a_public],
                search_service=search_service_with_mock_embedding,
            )
            client = TestClient(app)
            resp = client.post(
                "/api/search", json={"query": "BM25 单路"}
            )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        # 仅 BM25 单路返回，3 条结果应全部进入最终响应
        assert body["total"] == 3
        chunk_ids = [r["chunk_id"] for r in body["results"]]
        assert chunk_ids == ["bm25_0", "bm25_1", "bm25_2"]


# ─── 边界场景 ─────────────────────────────────────────────────────────


class TestEdgeCaseQueries:
    """空白/超长/特殊字符查询都不应破坏管线。"""

    def test_whitespace_only_query_does_not_crash(
        self,
        search_service_with_mock_embedding: SearchService,
        user_a: uuid.UUID,
        space_a_public: str,
    ) -> None:
        """``query=" "`` 通过 ``min_length=1`` 校验，但服务层不应崩溃。

        关键词抽取会得到空列表，``_generate_highlight`` 走回退分支输出
        chunk 开头 200 字符。
        """
        os_hits = [
            _make_os_hit(
                chunk_id="c1",
                document_id="d1",
                space_id=space_a_public,
                content="任意命中的内容",
                score=2.0,
            )
        ]
        os_client = MagicMock()
        os_client.search.return_value = _opensearch_response(os_hits)
        qdrant_client = MagicMock()
        qdrant_client.search.return_value = []

        with patch(
            "app.core.opensearch.get_opensearch_client", return_value=os_client
        ), patch(
            "app.core.qdrant.get_qdrant_client", return_value=qdrant_client
        ):
            app = _build_app(
                user_id=user_a,
                allowed_space_ids=[space_a_public],
                search_service=search_service_with_mock_embedding,
            )
            client = TestClient(app)
            resp = client.post("/api/search", json={"query": "   "})

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        # 高亮回退到 chunk 开头，不应包含 <mark>
        assert HIGHLIGHT_MARK_OPEN not in body["results"][0]["highlight"]

    def test_max_length_query_500_chars(
        self,
        search_service_with_mock_embedding: SearchService,
        user_a: uuid.UUID,
        space_a_public: str,
    ) -> None:
        """500 字符是上限，应被接受并完整透传给 OpenSearch。"""
        os_client = MagicMock()
        os_client.search.return_value = _opensearch_response([])
        qdrant_client = MagicMock()
        qdrant_client.search.return_value = []

        # "复合搜索引擎" 6 字符 × 84 = 504，截到 500
        long_query = ("复合搜索引擎" * 84)[:500]
        assert len(long_query) == 500

        with patch(
            "app.core.opensearch.get_opensearch_client", return_value=os_client
        ), patch(
            "app.core.qdrant.get_qdrant_client", return_value=qdrant_client
        ):
            app = _build_app(
                user_id=user_a,
                allowed_space_ids=[space_a_public],
                search_service=search_service_with_mock_embedding,
            )
            client = TestClient(app)
            resp = client.post("/api/search", json={"query": long_query})

        assert resp.status_code == 200
        # OpenSearch 收到的查询应未被截断
        passed_body = os_client.search.call_args.kwargs["body"]
        assert (
            passed_body["query"]["bool"]["must"][0]["multi_match"]["query"]
            == long_query
        )

    def test_special_characters_query_supported(
        self,
        search_service_with_mock_embedding: SearchService,
        user_a: uuid.UUID,
        space_a_public: str,
    ) -> None:
        """引号、括号、emoji、CJK 等特殊字符不会破坏管线。"""
        special_query = '"知识库" (BM25) 🔍 — 复合检索 & 测试'

        os_hits = [
            _make_os_hit(
                chunk_id="c1",
                document_id="d1",
                space_id=space_a_public,
                content="知识库的复合检索包含 BM25 与向量召回",
                score=3.0,
            )
        ]
        os_client = MagicMock()
        os_client.search.return_value = _opensearch_response(os_hits)
        qdrant_client = MagicMock()
        qdrant_client.search.return_value = []

        with patch(
            "app.core.opensearch.get_opensearch_client", return_value=os_client
        ), patch(
            "app.core.qdrant.get_qdrant_client", return_value=qdrant_client
        ):
            app = _build_app(
                user_id=user_a,
                allowed_space_ids=[space_a_public],
                search_service=search_service_with_mock_embedding,
            )
            client = TestClient(app)
            resp = client.post("/api/search", json={"query": special_query})

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert len(body["results"][0]["highlight"]) <= HIGHLIGHT_MAX_CHARS
        # 仍能命中关键词「知识库」「BM25」
        highlight = body["results"][0]["highlight"]
        assert "知识库" in highlight or "BM25" in highlight

    def test_long_content_highlight_truncated_to_200(
        self,
        search_service_with_mock_embedding: SearchService,
        user_a: uuid.UUID,
        space_a_public: str,
    ) -> None:
        """超长 chunk 内容（>200 字符）应被裁切到不超过 200 字符。"""
        long_content = "前置文本" * 100 + "关键词命中" + "后置文本" * 100
        os_hits = [
            _make_os_hit(
                chunk_id="c1",
                document_id="d1",
                space_id=space_a_public,
                content=long_content,
                score=5.0,
            )
        ]
        os_client = MagicMock()
        os_client.search.return_value = _opensearch_response(os_hits)
        qdrant_client = MagicMock()
        qdrant_client.search.return_value = []

        with patch(
            "app.core.opensearch.get_opensearch_client", return_value=os_client
        ), patch(
            "app.core.qdrant.get_qdrant_client", return_value=qdrant_client
        ):
            app = _build_app(
                user_id=user_a,
                allowed_space_ids=[space_a_public],
                search_service=search_service_with_mock_embedding,
            )
            client = TestClient(app)
            resp = client.post("/api/search", json={"query": "关键词"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        highlight = body["results"][0]["highlight"]
        assert len(highlight) <= HIGHLIGHT_MAX_CHARS
        # 仍包含 mark 包裹的「关键词」命中
        assert "<mark>" in highlight


# ─── 全局 5 秒超时与单路 3 秒超时联动 ────────────────────────────────


class TestTotalTimeoutInteraction:
    """全局 5 秒超时是单路超时之上的兜底。"""

    def test_total_timeout_returns_504_when_overall_search_too_slow(
        self,
        search_service_with_mock_embedding: SearchService,
        user_a: uuid.UUID,
        space_a_public: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``SearchService.search`` 整体执行超过 ``SEARCH_TOTAL_TIMEOUT`` 时
        路由层应返回 504。"""
        from app.api import search as search_module

        # 把整体超时压缩到 0.05 秒，触发 504 而无须真等 5 秒
        monkeypatch.setattr(search_module, "SEARCH_TOTAL_TIMEOUT", 0.05)

        # 让 SearchService.search 长时间挂起
        async def _slow_search(**_kwargs):
            await asyncio.sleep(2.0)
            raise AssertionError("不应执行到这里")

        slow_service = AsyncMock(spec=SearchService)
        slow_service.search.side_effect = _slow_search

        app = _build_app(
            user_id=user_a,
            allowed_space_ids=[space_a_public],
            search_service=slow_service,
        )
        client = TestClient(app)
        resp = client.post("/api/search", json={"query": "慢查询"})

        assert resp.status_code == 504
        body = resp.json()
        assert body["error"]["code"] == "SearchTimeout"
