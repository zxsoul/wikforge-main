"""搜索端到端测试（任务 25.3）。

目标：用多种查询类型验证 ``POST /api/search`` 返回的相关性。覆盖：

- 完全匹配（关键词在 chunk content 中出现）
- 语义匹配（同义词、措辞不同但语义相近）
- 多语言混合（中文 + 英文）
- 短查询 / 长查询
- 排序：相关性高的应排在前面

策略：仍以 mock 后端的方式注入"已索引的语料"，让 SearchService 真实跑
RRF + Cross-Encoder fallback；通过控制 BM25 / Dense 的相对分数来模拟
"哪些 chunk 应该排在前面"。

Validates: Requirements 6
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.auth import get_current_user
from app.api.search import get_search_service
from app.api.search import router as search_router
from app.core.database import get_db
from app.core.exceptions import register_exception_handlers
from app.services.search_service import SearchService


pytestmark = pytest.mark.integration


# ─── 小语料库 ──────────────────────────────────────────────────────────


def _corpus(space_id: str) -> list[dict]:
    """构造一个 4 文档的小语料库，覆盖语义/关键词/混合三种命中类型。"""
    return [
        {
            "chunk_id": "rag-1",
            "document_id": "doc-rag",
            "space_id": space_id,
            "chunk_index": 0,
            "title_chain": "RAG 概述",
            "source_file": "rag.pdf",
            "page_number": 1,
            "content": "RAG (Retrieval-Augmented Generation) 通过检索增强生成，把外部知识库注入大模型。",
            "allowed_user_ids": [],
        },
        {
            "chunk_id": "vec-1",
            "document_id": "doc-vec",
            "space_id": space_id,
            "chunk_index": 0,
            "title_chain": "向量检索",
            "source_file": "vector.md",
            "page_number": 1,
            "content": "Dense embedding 用于语义检索，常见模型包括 BGE 与 E5。",
            "allowed_user_ids": [],
        },
        {
            "chunk_id": "bm25-1",
            "document_id": "doc-bm25",
            "space_id": space_id,
            "chunk_index": 0,
            "title_chain": "BM25",
            "source_file": "bm25.md",
            "page_number": 1,
            "content": "BM25 是经典稀疏关键词检索算法，依赖 IDF 与文档长度归一。",
            "allowed_user_ids": [],
        },
        {
            "chunk_id": "off-topic",
            "document_id": "doc-off",
            "space_id": space_id,
            "chunk_index": 0,
            "title_chain": "其它主题",
            "source_file": "off.md",
            "page_number": 1,
            "content": "今天天气不错，适合去公园散步。",
            "allowed_user_ids": [],
        },
    ]


def _build_app(*, user_id: uuid.UUID, space_id: str) -> tuple[FastAPI, MagicMock, MagicMock]:
    embedding_service = AsyncMock()
    embedding_service.embed_query = AsyncMock(
        return_value=MagicMock(
            chunk_id="query",
            dense_vector=[0.01] * 1024,
            sparse_indices=[1, 2, 3],
            sparse_values=[0.5, 0.3, 0.2],
        )
    )

    search_service = SearchService(embedding_service=embedding_service)

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(search_router)

    fake_user = MagicMock()
    fake_user.id = user_id

    async def _override_user():
        return fake_user

    db_session = AsyncMock()
    db_result = MagicMock()
    db_result.scalars.return_value.all.return_value = [space_id]
    db_session.execute = AsyncMock(return_value=db_result)

    async def _override_db():
        yield db_session

    app.dependency_overrides[get_current_user] = _override_user
    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_search_service] = lambda: search_service

    os_client = MagicMock()
    qdrant_client = MagicMock()
    return app, os_client, qdrant_client


def _os_response_for(corpus: list[dict], score_map: dict[str, float]) -> dict:
    hits = []
    for c in corpus:
        score = score_map.get(c["chunk_id"])
        if score is None:
            continue
        hits.append({"_id": c["chunk_id"], "_score": score, "_source": c})
    hits.sort(key=lambda h: h["_score"], reverse=True)
    return {
        "took": 1,
        "timed_out": False,
        "hits": {
            "total": {"value": len(hits), "relation": "eq"},
            "hits": hits,
        },
    }


def _qdrant_points_for(corpus: list[dict], score_map: dict[str, float]) -> list:
    pts = []
    for c in corpus:
        score = score_map.get(c["chunk_id"])
        if score is None:
            continue
        pts.append(
            SimpleNamespace(
                id=c["chunk_id"],
                score=score,
                payload={k: v for k, v in c.items() if k != "chunk_id"},
            )
        )
    pts.sort(key=lambda p: p.score, reverse=True)
    return pts


# ─── 测试 ─────────────────────────────────────────────────────────────


class TestSearchRelevance:
    """多种查询类型下，相关 chunk 应排在前面，无关 chunk 应排在后或被过滤。"""

    def test_keyword_query_ranks_keyword_match_first(self) -> None:
        """关键词查询「BM25」应让 bm25-1 排第一。"""
        user_id = uuid.uuid4()
        space_id = str(uuid.uuid4())
        corpus = _corpus(space_id)

        app, os_client, qdrant_client = _build_app(user_id=user_id, space_id=space_id)

        # BM25 命中：bm25-1 高分；rag-1 也出现 BM25 字面但分数低
        os_client.search.return_value = _os_response_for(
            corpus, {"bm25-1": 9.0, "rag-1": 2.5}
        )
        # Dense / Sparse 给同样的次序
        qdrant_client.search.return_value = _qdrant_points_for(
            corpus, {"bm25-1": 0.85, "rag-1": 0.55}
        )

        with patch("app.core.opensearch.get_opensearch_client", return_value=os_client), \
                patch("app.core.qdrant.get_qdrant_client", return_value=qdrant_client):
            client = TestClient(app)
            resp = client.post("/api/search", json={"query": "BM25"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 1
        assert body["results"][0]["chunk_id"] == "bm25-1"

    def test_semantic_query_ranks_semantic_match_first(self) -> None:
        """语义查询「语义检索的方法」应让 vec-1 排第一（即使没有字面命中）。"""
        user_id = uuid.uuid4()
        space_id = str(uuid.uuid4())
        corpus = _corpus(space_id)

        app, os_client, qdrant_client = _build_app(user_id=user_id, space_id=space_id)

        # BM25 不命中（无字面）；Dense 给 vec-1 高分
        os_client.search.return_value = _os_response_for(corpus, {})
        qdrant_client.search.return_value = _qdrant_points_for(
            corpus, {"vec-1": 0.92, "rag-1": 0.61, "bm25-1": 0.40}
        )

        with patch("app.core.opensearch.get_opensearch_client", return_value=os_client), \
                patch("app.core.qdrant.get_qdrant_client", return_value=qdrant_client):
            client = TestClient(app)
            resp = client.post("/api/search", json={"query": "语义检索的方法"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 1
        assert body["results"][0]["chunk_id"] == "vec-1"

    def test_mixed_lang_query_finds_relevant_chunks(self) -> None:
        """中英文混合查询「RAG 检索增强」应命中 rag-1。"""
        user_id = uuid.uuid4()
        space_id = str(uuid.uuid4())
        corpus = _corpus(space_id)

        app, os_client, qdrant_client = _build_app(user_id=user_id, space_id=space_id)

        os_client.search.return_value = _os_response_for(
            corpus, {"rag-1": 8.5, "vec-1": 3.0}
        )
        qdrant_client.search.return_value = _qdrant_points_for(
            corpus, {"rag-1": 0.95, "vec-1": 0.65}
        )

        with patch("app.core.opensearch.get_opensearch_client", return_value=os_client), \
                patch("app.core.qdrant.get_qdrant_client", return_value=qdrant_client):
            client = TestClient(app)
            resp = client.post("/api/search", json={"query": "RAG 检索增强"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["results"][0]["chunk_id"] == "rag-1"

    def test_off_topic_query_returns_low_or_no_relevance(self) -> None:
        """无关查询「公园散步」返回的 off-topic 应排在所有相关结果之后或不出现。"""
        user_id = uuid.uuid4()
        space_id = str(uuid.uuid4())
        corpus = _corpus(space_id)

        app, os_client, qdrant_client = _build_app(user_id=user_id, space_id=space_id)

        # 仅 off-topic 文档命中
        os_client.search.return_value = _os_response_for(
            corpus, {"off-topic": 4.0}
        )
        qdrant_client.search.return_value = _qdrant_points_for(
            corpus, {"off-topic": 0.30}
        )

        with patch("app.core.opensearch.get_opensearch_client", return_value=os_client), \
                patch("app.core.qdrant.get_qdrant_client", return_value=qdrant_client):
            client = TestClient(app)
            resp = client.post("/api/search", json={"query": "公园散步"})

        assert resp.status_code == 200
        body = resp.json()
        # 仅命中 off-topic
        assert {r["chunk_id"] for r in body["results"]} == {"off-topic"}

    def test_pagination_preserves_ordering(self) -> None:
        """跨页结果应保留全局排序。"""
        user_id = uuid.uuid4()
        space_id = str(uuid.uuid4())
        corpus = _corpus(space_id)

        app, os_client, qdrant_client = _build_app(user_id=user_id, space_id=space_id)

        os_client.search.return_value = _os_response_for(
            corpus,
            {"rag-1": 10.0, "vec-1": 8.0, "bm25-1": 6.0, "off-topic": 4.0},
        )
        qdrant_client.search.return_value = _qdrant_points_for(corpus, {})

        with patch("app.core.opensearch.get_opensearch_client", return_value=os_client), \
                patch("app.core.qdrant.get_qdrant_client", return_value=qdrant_client):
            client = TestClient(app)
            page1 = client.post(
                "/api/search", json={"query": "检索", "page": 1, "page_size": 2}
            ).json()
            page2 = client.post(
                "/api/search", json={"query": "检索", "page": 2, "page_size": 2}
            ).json()

        ids_p1 = [r["chunk_id"] for r in page1["results"]]
        ids_p2 = [r["chunk_id"] for r in page2["results"]]
        assert ids_p1 == ["rag-1", "vec-1"]
        assert ids_p2 == ["bm25-1", "off-topic"]
