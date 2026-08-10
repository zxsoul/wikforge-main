"""性能测试（任务 25.8）。

验证关键操作的延迟符合需求文档的约束：
- 搜索 5 秒内返回（Requirements 6）
- RAG 首 token 5 秒内返回（Requirements 8）
- 文档上传 100MB 5 分钟内完成（Requirements 1）
- 权限判定 50ms（Requirements 10）

策略：
- 用 mock 后端把外部服务延迟降为 0，仅测量管线本身的开销
- 通过 ``pytest.mark.benchmark`` 标记，可按需 ``pytest -m benchmark`` 跑
- 不引入 locust 等重依赖，仅用 ``time.perf_counter`` 简单计时

注意：在没有真实容器/CPU 资源时，本测试仅验证"管线本身不引入额外
显著开销"。真实端到端 SLO 的验证应在压测环境用 locust 或 k6 完成。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.benchmark]


# ─── 搜索性能 ─────────────────────────────────────────────────────────


class TestSearchLatency:
    """``POST /api/search`` 应在 5 秒内返回（mock 后端，仅测管线本身）。"""

    def test_search_returns_within_5s(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from app.api.auth import get_current_user
        from app.api.search import get_search_service
        from app.api.search import router as search_router
        from app.core.database import get_db
        from app.core.exceptions import register_exception_handlers
        from app.services.search_service import SearchService

        user_id = uuid.uuid4()
        space_id = str(uuid.uuid4())

        # 100 条命中模拟典型查询规模
        os_hits = [
            {
                "_id": f"c{i}",
                "_score": 5.0 - i * 0.01,
                "_source": {
                    "chunk_id": f"c{i}",
                    "document_id": f"doc-{i}",
                    "space_id": space_id,
                    "chunk_index": 0,
                    "title_chain": "x",
                    "source_file": "x.pdf",
                    "page_number": 1,
                    "content": f"内容 {i}",
                    "allowed_user_ids": [],
                },
            }
            for i in range(100)
        ]

        os_client = MagicMock()
        os_client.search.return_value = {
            "took": 1,
            "timed_out": False,
            "hits": {"total": {"value": 100, "relation": "eq"}, "hits": os_hits},
        }

        qdrant_client = MagicMock()
        qdrant_client.search.return_value = []

        embedding_service = AsyncMock()
        embedding_service.embed_query = AsyncMock(
            return_value=MagicMock(
                chunk_id="q",
                dense_vector=[0.01] * 1024,
                sparse_indices=[1],
                sparse_values=[0.5],
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

        with patch("app.core.opensearch.get_opensearch_client", return_value=os_client), \
                patch("app.core.qdrant.get_qdrant_client", return_value=qdrant_client):
            client = TestClient(app)

            start = time.perf_counter()
            resp = client.post("/api/search", json={"query": "性能测试"})
            elapsed = time.perf_counter() - start

        assert resp.status_code == 200
        # SLO：5 秒内返回
        assert elapsed < 5.0, f"搜索耗时 {elapsed:.2f}s 超过 5s SLO"


# ─── RAG 首 token 延迟 ─────────────────────────────────────────────────


class TestRAGFirstTokenLatency:
    """RAG 流式首 token 应在 5 秒内（Requirements 8.3）。"""

    @pytest.mark.asyncio
    async def test_first_token_within_5s(self) -> None:
        import pytest_asyncio  # noqa: F401  确保 plugin 启用

        from app.services.conversation_service import ConversationService
        from app.services.rag_service import RAGService, STREAM_EVENT_TOKEN
        from app.services.search_service import SearchResponse, SearchResult

        fakeredis = pytest.importorskip("fakeredis")
        redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        try:
            conv = ConversationService(redis_client=redis)

            results = [
                SearchResult(
                    chunk_id="c-1",
                    document_id="d-1",
                    chunk_index=0,
                    title_chain="x",
                    source_file="x.pdf",
                    page_number=1,
                    score=0.9,
                    highlight="片段",
                )
            ]
            search = AsyncMock()
            search.search = AsyncMock(
                return_value=SearchResponse(
                    results=results, total=1, page=1, page_size=1
                )
            )

            async def _stream(*_a, **_kw):
                yield "首"
                yield "token"

            gateway = MagicMock()
            gateway.stream = _stream

            service = RAGService(
                search_service=search,
                llm_gateway=gateway,
                conversation_service=conv,
            )

            start = time.perf_counter()
            first_token_at = None
            async for ev in service.answer_stream(
                query="性能问题",
                user_id="u",
                allowed_space_ids=["s"],
                conversation_id="conv-perf",
            ):
                if ev.event == STREAM_EVENT_TOKEN and first_token_at is None:
                    first_token_at = time.perf_counter() - start
                    break

            assert first_token_at is not None
            # SLO：5 秒
            assert first_token_at < 5.0, f"首 token 耗时 {first_token_at:.3f}s"
        finally:
            await redis.aclose()


# ─── 权限判定延迟 ─────────────────────────────────────────────────────


class TestPermissionLatency:
    """``check_access`` 应在 50ms 内（Requirements 10.1，命中缓存）。"""

    @pytest.mark.asyncio
    async def test_permission_check_under_50ms(self) -> None:
        from app.models.permission import AccessLevel, ResourceType
        from app.services.permission_service import Action, PermissionService

        # 缓存命中场景：让 Redis 返回已缓存的 access level
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=AccessLevel.read.value)
        mock_redis.setex = AsyncMock()

        mock_db = AsyncMock()

        service = PermissionService(db=mock_db, redis=mock_redis)

        user_id = uuid.uuid4()
        resource_id = uuid.uuid4()

        # 预热：跑一次免去模块加载的影响
        await service.check_access(
            user_id=user_id,
            resource_id=resource_id,
            resource_type=ResourceType.space,
            action=Action.read,
        )

        n = 100
        start = time.perf_counter()
        for _ in range(n):
            await service.check_access(
                user_id=user_id,
                resource_id=resource_id,
                resource_type=ResourceType.space,
                action=Action.read,
            )
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / n) * 1000.0

        # SLO：每次平均 < 50ms
        assert avg_ms < 50.0, f"权限判定平均 {avg_ms:.2f}ms 超过 50ms SLO"


# ─── 文档上传吞吐 ─────────────────────────────────────────────────────


class TestUploadThroughput:
    """100MB 文件上传到 MinIO 不应阻塞主流程 5 分钟。

    本测试仅模拟 MinIO 客户端瞬时返回，验证 UploadService 自身没有引入
    额外的同步阻塞或不必要的 IO 循环。真实 100MB 上传速度由 MinIO 与
    网络带宽决定。
    """

    def test_upload_pipeline_overhead_under_5min(self) -> None:
        from io import BytesIO

        from fastapi import UploadFile
        from starlette.datastructures import Headers

        from app.services.upload_service import UploadService

        mock_db = AsyncMock()
        mock_db.add = MagicMock()
        mock_db.flush = AsyncMock()
        mock_db.refresh = AsyncMock()

        async def _refresh(doc):
            doc.id = uuid.uuid4()

        mock_db.refresh.side_effect = _refresh

        # 5MB 模拟（100MB 在 CI 中过慢，且我们仅测量管线开销）
        size = 5 * 1024 * 1024
        upload_file = UploadFile(
            file=BytesIO(b"%PDF-1.4\n" + b"x" * (size - 9)),
            filename="big.pdf",
            headers=Headers({"content-type": "application/pdf"}),
        )

        service = UploadService(db=mock_db)

        with patch("app.services.upload_service.get_minio_client") as mock_factory, \
                patch("app.services.upload_service.ensure_bucket_exists"), \
                patch.object(service, "_init_redis_status", AsyncMock()):
            mock_factory.return_value = MagicMock()

            start = time.perf_counter()
            asyncio.run(
                service.upload_files(
                    files=[upload_file],
                    space_id=uuid.uuid4(),
                    folder_id=None,
                    uploaded_by=uuid.uuid4(),
                )
            )
            elapsed = time.perf_counter() - start

        # SLO：5MB 文件管线开销应远低于 5 分钟（我们设 5 秒兜底）
        assert elapsed < 5.0, f"上传管线开销 {elapsed:.2f}s 超过 5s 阈值"
