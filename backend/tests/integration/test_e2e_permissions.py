"""权限端到端测试（任务 25.5）。

覆盖：设置权限 → 搜索 → 验证无权限文档不出现。

策略：
- 通过 Search API 路由 + ``allowed_space_ids`` 过滤验证 Pre-Filtering 链路
- 模拟 OpenSearch / Qdrant 服务端按权限 filter 过滤命中结果
- 验证三种隔离：空间私有 / 显式分享 / 完全无权限

Validates: Requirements 10
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


# ─── Helpers（与 test_search_integration 同模式） ────────────────────


def _make_hit(*, chunk_id: str, space_id: str, allowed_user_ids: list[str]) -> dict:
    return {
        "_id": chunk_id,
        "_score": 5.0,
        "_source": {
            "chunk_id": chunk_id,
            "document_id": f"doc-{chunk_id}",
            "space_id": space_id,
            "chunk_index": 0,
            "title_chain": "测试",
            "source_file": "demo.pdf",
            "page_number": 1,
            "content": f"内容 {chunk_id}",
            "allowed_user_ids": allowed_user_ids,
        },
    }


def _make_point(*, chunk_id: str, space_id: str, allowed_user_ids: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        id=chunk_id,
        score=0.85,
        payload={
            "document_id": f"doc-{chunk_id}",
            "space_id": space_id,
            "chunk_index": 0,
            "title_chain": "测试",
            "source_file": "demo.pdf",
            "page_number": 1,
            "content": f"内容 {chunk_id}",
            "allowed_user_ids": allowed_user_ids,
        },
    )


def _build_app(*, user_id: uuid.UUID, allowed_space_ids: list[str]) -> FastAPI:
    embedding_service = AsyncMock()
    embedding_service.embed_query = AsyncMock(
        return_value=MagicMock(
            chunk_id="query",
            dense_vector=[0.01] * 1024,
            sparse_indices=[1, 2],
            sparse_values=[0.5, 0.3],
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
    db_result.scalars.return_value.all.return_value = list(allowed_space_ids)
    db_session.execute = AsyncMock(return_value=db_result)

    async def _override_db():
        yield db_session

    app.dependency_overrides[get_current_user] = _override_user
    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_search_service] = lambda: search_service
    return app


# ─── 测试 ─────────────────────────────────────────────────────────────


class TestPermissionEndToEnd:
    """权限设置后，搜索结果应严格按权限过滤。"""

    def test_user_without_space_access_cannot_see_chunks(self) -> None:
        """用户对空间 X 无权限时，X 里的 chunk 不应出现在搜索结果中。"""
        user_a = uuid.uuid4()
        user_b = uuid.uuid4()
        space_public = str(uuid.uuid4())
        space_private = str(uuid.uuid4())

        all_hits = [
            _make_hit(chunk_id="public", space_id=space_public, allowed_user_ids=[]),
            _make_hit(
                chunk_id="private",
                space_id=space_private,
                allowed_user_ids=[str(user_b)],
            ),
        ]
        all_points = [
            _make_point(chunk_id="public", space_id=space_public, allowed_user_ids=[]),
            _make_point(
                chunk_id="private",
                space_id=space_private,
                allowed_user_ids=[str(user_b)],
            ),
        ]

        os_client = MagicMock()
        qdrant_client = MagicMock()

        # 模拟 OpenSearch 服务端按 filter 过滤
        def _os_search(*, index, body, **_):
            should = body["query"]["bool"]["filter"][0]["bool"]["should"]
            uid = next(c["term"]["allowed_user_ids"] for c in should if "term" in c)
            sids = next(c["terms"]["space_id"] for c in should if "terms" in c)
            kept = [
                h
                for h in all_hits
                if uid in (h["_source"]["allowed_user_ids"] or [])
                or h["_source"]["space_id"] in sids
            ]
            return {
                "took": 1,
                "timed_out": False,
                "hits": {
                    "total": {"value": len(kept), "relation": "eq"},
                    "hits": kept,
                },
            }

        os_client.search.side_effect = _os_search

        # 模拟 Qdrant 服务端按 filter 过滤
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
                    if cond.key == "allowed_user_ids" and isinstance(cond.match, MatchValue):
                        uid = cond.match.value
                    elif cond.key == "space_id" and isinstance(cond.match, MatchAny):
                        sids = list(cond.match.any)

            qv = kwargs["query_vector"]
            kept = [
                p
                for p in all_points
                if uid in (p.payload.get("allowed_user_ids") or [])
                or p.payload.get("space_id") in sids
            ]
            if isinstance(qv, NamedVector):
                return kept
            if isinstance(qv, NamedSparseVector):
                return []
            return []

        qdrant_client.search.side_effect = _qdrant_search

        # 用户 A 仅对 space_public 有访问权
        app = _build_app(user_id=user_a, allowed_space_ids=[space_public])

        with patch("app.core.opensearch.get_opensearch_client", return_value=os_client), \
                patch("app.core.qdrant.get_qdrant_client", return_value=qdrant_client):
            client = TestClient(app)
            resp = client.post("/api/search", json={"query": "内容"})

        assert resp.status_code == 200
        body = resp.json()
        chunk_ids = {r["chunk_id"] for r in body["results"]}
        assert "public" in chunk_ids
        assert "private" not in chunk_ids, "无权限文档不能出现在搜索结果"

    def test_explicit_user_share_grants_access(self) -> None:
        """文档的 ``allowed_user_ids`` 显式包含用户时，即便对该空间没有访问权也能搜到。

        前置：用户在 space_a 有访问权（满足 SearchService 的非空空间短路），
        但实际感兴趣的内容在 space_b。space_b 中只有 ``shared`` 显式分享给了
        用户，``locked`` 没有。预期：能搜到 shared，搜不到 locked。
        """
        user_a = uuid.uuid4()
        space_a = str(uuid.uuid4())  # 用户有访问权但里面没内容
        space_b = str(uuid.uuid4())  # 用户无访问权,但有显式分享

        all_hits = [
            _make_hit(
                chunk_id="shared",
                space_id=space_b,
                allowed_user_ids=[str(user_a)],
            ),
            _make_hit(chunk_id="locked", space_id=space_b, allowed_user_ids=[]),
        ]
        all_points = [
            _make_point(
                chunk_id="shared",
                space_id=space_b,
                allowed_user_ids=[str(user_a)],
            ),
            _make_point(chunk_id="locked", space_id=space_b, allowed_user_ids=[]),
        ]

        os_client = MagicMock()
        qdrant_client = MagicMock()

        def _os_search(*, index, body, **_):
            should = body["query"]["bool"]["filter"][0]["bool"]["should"]
            uid = next(c["term"]["allowed_user_ids"] for c in should if "term" in c)
            sids = next(c["terms"]["space_id"] for c in should if "terms" in c)
            kept = [
                h
                for h in all_hits
                if uid in (h["_source"]["allowed_user_ids"] or [])
                or h["_source"]["space_id"] in sids
            ]
            return {
                "took": 1,
                "timed_out": False,
                "hits": {
                    "total": {"value": len(kept), "relation": "eq"},
                    "hits": kept,
                },
            }

        os_client.search.side_effect = _os_search

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
                    if cond.key == "allowed_user_ids" and isinstance(cond.match, MatchValue):
                        uid = cond.match.value
                    elif cond.key == "space_id" and isinstance(cond.match, MatchAny):
                        sids = list(cond.match.any)

            qv = kwargs["query_vector"]
            kept = [
                p
                for p in all_points
                if uid in (p.payload.get("allowed_user_ids") or [])
                or p.payload.get("space_id") in sids
            ]
            if isinstance(qv, NamedVector):
                return kept
            if isinstance(qv, NamedSparseVector):
                return []
            return []

        qdrant_client.search.side_effect = _qdrant_search

        # 用户 A 至少在某个空间有访问权（SearchService 才会进行检索）
        app = _build_app(user_id=user_a, allowed_space_ids=[space_a])

        with patch("app.core.opensearch.get_opensearch_client", return_value=os_client), \
                patch("app.core.qdrant.get_qdrant_client", return_value=qdrant_client):
            client = TestClient(app)
            resp = client.post("/api/search", json={"query": "内容"})

        assert resp.status_code == 200
        chunk_ids = {r["chunk_id"] for r in resp.json()["results"]}
        assert "shared" in chunk_ids
        assert "locked" not in chunk_ids

    def test_no_permission_user_returns_empty_results(self) -> None:
        """完全无权限用户搜索应返回空，且不发起后端调用（短路）。"""
        user_a = uuid.uuid4()

        os_client = MagicMock()
        qdrant_client = MagicMock()

        app = _build_app(user_id=user_a, allowed_space_ids=[])

        with patch("app.core.opensearch.get_opensearch_client", return_value=os_client), \
                patch("app.core.qdrant.get_qdrant_client", return_value=qdrant_client):
            client = TestClient(app)
            resp = client.post("/api/search", json={"query": "任意"})

        assert resp.status_code == 200
        assert resp.json()["total"] == 0
        # SearchService 在 allowed_space_ids 为空时应短路
        assert os_client.search.call_count == 0
        assert qdrant_client.search.call_count == 0
