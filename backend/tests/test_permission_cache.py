"""Redis 权限缓存测试 (任务 4.7)。

覆盖：
- 缓存命中时不查询 DB
- 缓存 miss 时查询 DB 并写入 ``ex=PERM_CACHE_TTL=300``
- 不存在的权限以 ``__none__`` 哨兵缓存，避免缓存穿透
- 主动失效：单 key 删除 + 模式扫描批量删除
- 通过 fakeredis 验证真实 TTL 设置（而非仅 mock 行为）

Validates: Requirements 10
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest

from app.models.permission import AccessLevel, ResourceType
from app.services.permission_service import (
    PERM_CACHE_KEY_PATTERN,
    PERM_CACHE_TTL,
    PermissionService,
)
from tests._permission_helpers import (
    permission_service,
    perm_db,
    perm_redis,
    scalar_one_or_none_result,
)


# ─── TTL 常量约束 ───────────────────────────────────────────────────


def test_ttl_is_5_minutes():
    """缓存 TTL 应为设计文档约定的 300 秒（5 分钟）。"""
    assert PERM_CACHE_TTL == 300


def test_cache_key_pattern_contains_user_and_space():
    """缓存键模式应同时包含 user_id 和 space_id 占位符。"""
    assert "{user_id}" in PERM_CACHE_KEY_PATTERN
    assert "{space_id}" in PERM_CACHE_KEY_PATTERN


# ─── 缓存命中 / miss 行为 ──────────────────────────────────────────


class TestCacheBehavior:
    @pytest.mark.asyncio
    async def test_cache_hit_does_not_query_db(
        self, permission_service, perm_db, perm_redis
    ):
        """缓存命中时不应触发 DB 查询。"""
        user_id = uuid.uuid4()
        space_id = uuid.uuid4()
        perm_redis.get = AsyncMock(return_value="write")
        perm_db.execute = AsyncMock()

        result = await permission_service._get_space_permission_cached(
            user_id, space_id
        )
        assert result == AccessLevel.write
        perm_db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_queries_db_and_sets_with_ttl(
        self, permission_service, perm_db, perm_redis
    ):
        """缓存 miss 时查询 DB，并以 300 秒 TTL 写回缓存。"""
        user_id = uuid.uuid4()
        space_id = uuid.uuid4()

        perm_redis.get = AsyncMock(return_value=None)
        perm_db.execute = AsyncMock(
            return_value=scalar_one_or_none_result(AccessLevel.read)
        )

        result = await permission_service._get_space_permission_cached(
            user_id, space_id
        )
        assert result == AccessLevel.read

        cache_key = PERM_CACHE_KEY_PATTERN.format(
            user_id=str(user_id), space_id=str(space_id)
        )
        perm_redis.set.assert_awaited_once_with(
            cache_key, "read", ex=PERM_CACHE_TTL
        )

    @pytest.mark.asyncio
    async def test_cache_stores_none_sentinel(
        self, permission_service, perm_db, perm_redis
    ):
        """DB 无权限记录时缓存 ``__none__`` 哨兵以避免缓存穿透。"""
        user_id = uuid.uuid4()
        space_id = uuid.uuid4()

        perm_redis.get = AsyncMock(return_value=None)
        perm_db.execute = AsyncMock(
            return_value=scalar_one_or_none_result(None)
        )

        result = await permission_service._get_space_permission_cached(
            user_id, space_id
        )
        assert result is None

        perm_redis.set.assert_awaited_once()
        args, kwargs = perm_redis.set.await_args
        assert args[1] == "__none__"
        assert kwargs.get("ex") == PERM_CACHE_TTL

    @pytest.mark.asyncio
    async def test_cache_sentinel_returns_none_without_db(
        self, permission_service, perm_db, perm_redis
    ):
        """缓存中的 ``__none__`` 应直接返回 None，不查询 DB。"""
        user_id = uuid.uuid4()
        space_id = uuid.uuid4()

        perm_redis.get = AsyncMock(return_value="__none__")
        perm_db.execute = AsyncMock()

        result = await permission_service._get_space_permission_cached(
            user_id, space_id
        )
        assert result is None
        perm_db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_garbage_value_falls_back_to_db(
        self, permission_service, perm_db, perm_redis
    ):
        """缓存中存在非法值时应安全降级到 DB 查询。"""
        user_id = uuid.uuid4()
        space_id = uuid.uuid4()

        perm_redis.get = AsyncMock(return_value="bogus")
        perm_db.execute = AsyncMock(
            return_value=scalar_one_or_none_result(AccessLevel.read)
        )

        result = await permission_service._get_space_permission_cached(
            user_id, space_id
        )
        assert result == AccessLevel.read


# ─── 主动失效 ──────────────────────────────────────────────────────


class TestCacheInvalidation:
    @pytest.mark.asyncio
    async def test_invalidate_single_key(
        self, permission_service, perm_redis
    ):
        """``_invalidate_space_cache`` 应只删除 (user, space) 对应的 key。"""
        user_id = uuid.uuid4()
        space_id = uuid.uuid4()

        await permission_service._invalidate_space_cache(user_id, space_id)

        expected = PERM_CACHE_KEY_PATTERN.format(
            user_id=str(user_id), space_id=str(space_id)
        )
        perm_redis.delete.assert_awaited_once_with(expected)

    @pytest.mark.asyncio
    async def test_invalidate_all_space_cache_scans_pattern(
        self, perm_db
    ):
        """``invalidate_all_space_cache`` 应按 pattern 扫描并逐个删除。"""
        space_id = uuid.uuid4()
        u1, u2 = uuid.uuid4(), uuid.uuid4()
        keys = [
            PERM_CACHE_KEY_PATTERN.format(user_id=str(u1), space_id=str(space_id)),
            PERM_CACHE_KEY_PATTERN.format(user_id=str(u2), space_id=str(space_id)),
        ]

        # 自定义 redis mock：scan_iter 返回上述 keys
        from unittest.mock import AsyncMock as _AM, MagicMock as _MM

        redis = _MM()
        redis.delete = _AM(return_value=1)

        async def _scan_iter(match=None):
            assert match == f"perm:user:*:space:{space_id}"
            for k in keys:
                yield k

        redis.scan_iter = _scan_iter

        service = PermissionService(db=perm_db, redis=redis)
        await service.invalidate_all_space_cache(space_id)

        assert redis.delete.await_count == 2


# ─── fakeredis 真实 TTL 验证 ───────────────────────────────────────


@pytest.mark.asyncio
async def test_real_ttl_is_set_via_fakeredis(perm_db):
    """通过 fakeredis 验证 ``set ex=300`` 真的被应用到 key 上。"""
    fakeredis = pytest.importorskip("fakeredis")
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    try:
        service = PermissionService(db=perm_db, redis=redis)
        user_id = uuid.uuid4()
        space_id = uuid.uuid4()

        # DB 返回 read
        perm_db.execute = AsyncMock(
            return_value=scalar_one_or_none_result(AccessLevel.read)
        )

        result = await service._get_space_permission_cached(user_id, space_id)
        assert result == AccessLevel.read

        cache_key = PERM_CACHE_KEY_PATTERN.format(
            user_id=str(user_id), space_id=str(space_id)
        )
        ttl = await redis.ttl(cache_key)
        # 允许 ±2s 偏差（fakeredis 立即返回 300）
        assert 295 <= ttl <= 300

        # 二次调用应命中缓存（不再调用 DB）
        perm_db.execute = AsyncMock(side_effect=AssertionError("不应再次查询 DB"))
        result2 = await service._get_space_permission_cached(user_id, space_id)
        assert result2 == AccessLevel.read
    finally:
        await redis.aclose()
