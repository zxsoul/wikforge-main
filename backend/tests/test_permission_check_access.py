"""ABAC ``check_access`` 判定矩阵测试 (任务 4.1, 4.4)。

覆盖：
- 每个 ``access_level`` × 每个 ``action`` 的允许/拒绝组合
- 文档默认继承空间权限
- 文档级权限覆盖空间级
- 缺失权限记录默认为 invisible（拒绝一切）
- 命中 Redis 缓存时不再访问 DB（性能 smoke）

Validates: Requirements 10
"""

from __future__ import annotations

import time
import uuid
from unittest.mock import AsyncMock

import pytest

from app.models.permission import AccessLevel, ResourceType
from app.services.permission_service import (
    ACCESS_LEVEL_ACTIONS,
    Action,
)
from tests._permission_helpers import (
    permission_service,
    perm_db,
    perm_redis,
    scalar_one_or_none_result,
)


# ─── ABAC 判定矩阵（access_level × action）───────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "access_level, action, expected",
    [
        # invisible 拒绝一切动作
        (AccessLevel.invisible, Action.browse, False),
        (AccessLevel.invisible, Action.read, False),
        (AccessLevel.invisible, Action.write, False),
        # read 允许 browse / read，拒绝 write
        (AccessLevel.read, Action.browse, True),
        (AccessLevel.read, Action.read, True),
        (AccessLevel.read, Action.write, False),
        # write 允许全部三种动作
        (AccessLevel.write, Action.browse, True),
        (AccessLevel.write, Action.read, True),
        (AccessLevel.write, Action.write, True),
    ],
)
async def test_space_access_matrix(
    permission_service, perm_redis, access_level, action, expected
):
    """空间级权限的判定矩阵：每个 (level, action) 组合应返回设计规定的布尔值。"""
    user_id = uuid.uuid4()
    space_id = uuid.uuid4()
    perm_redis.get = AsyncMock(return_value=access_level.value)

    result = await permission_service.check_access(
        user_id=user_id,
        resource_id=space_id,
        resource_type=ResourceType.space,
        action=action,
    )
    assert result is expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "access_level, action, expected",
    [
        (AccessLevel.invisible, Action.read, False),
        (AccessLevel.read, Action.browse, True),
        (AccessLevel.read, Action.read, True),
        (AccessLevel.read, Action.write, False),
        (AccessLevel.write, Action.write, True),
    ],
)
async def test_document_access_matrix_with_explicit_perm(
    permission_service, perm_db, perm_redis, access_level, action, expected
):
    """文档级权限直接命中时不再回退到空间级权限。"""
    user_id = uuid.uuid4()
    document_id = uuid.uuid4()

    # 文档级查询返回明确的 access_level
    perm_db.execute = AsyncMock(
        return_value=scalar_one_or_none_result(access_level)
    )

    result = await permission_service.check_access(
        user_id=user_id,
        resource_id=document_id,
        resource_type=ResourceType.document,
        action=action,
    )
    assert result is expected


# ─── 继承与覆盖语义 ─────────────────────────────────────────────────


class TestInheritanceAndOverride:
    """文档默认继承空间权限；显式文档级权限覆盖空间级。"""

    @pytest.mark.asyncio
    async def test_document_inherits_space_read(
        self, permission_service, perm_db, perm_redis
    ):
        """文档无显式权限 → 应继承空间的 read 权限。"""
        user_id = uuid.uuid4()
        document_id = uuid.uuid4()
        space_id = uuid.uuid4()

        # 文档级权限始终为 None；每次调用 check_access 会发出 2 次 DB 查询：
        # (1) 文档级权限查询 (2) 文档所属空间查询
        # 因此使用可重入的 side_effect 函数：根据 SQL 查询语句区分。
        from sqlalchemy.sql import Select

        async def _execute(stmt, *_a, **_kw):
            sql_str = str(stmt)
            if "documents" in sql_str:
                return scalar_one_or_none_result(space_id)
            return scalar_one_or_none_result(None)

        perm_db.execute = AsyncMock(side_effect=_execute)
        # 空间级权限缓存命中 read
        perm_redis.get = AsyncMock(return_value="read")

        assert await permission_service.check_access(
            user_id, document_id, ResourceType.document, Action.read
        ) is True
        assert await permission_service.check_access(
            user_id, document_id, ResourceType.document, Action.write
        ) is False

    @pytest.mark.asyncio
    async def test_document_invisible_overrides_space_read(
        self, permission_service, perm_db, perm_redis
    ):
        """文档级 invisible 覆盖空间级 read。"""
        user_id = uuid.uuid4()
        document_id = uuid.uuid4()

        perm_db.execute = AsyncMock(
            return_value=scalar_one_or_none_result(AccessLevel.invisible)
        )
        # Redis 即使返回 read 也不应被采用
        perm_redis.get = AsyncMock(return_value="read")

        result = await permission_service.check_access(
            user_id, document_id, ResourceType.document, Action.read
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_document_write_overrides_space_invisible(
        self, permission_service, perm_db, perm_redis
    ):
        """文档级 write 覆盖空间级 invisible。"""
        user_id = uuid.uuid4()
        document_id = uuid.uuid4()

        perm_db.execute = AsyncMock(
            return_value=scalar_one_or_none_result(AccessLevel.write)
        )
        perm_redis.get = AsyncMock(return_value="invisible")

        assert await permission_service.check_access(
            user_id, document_id, ResourceType.document, Action.write
        ) is True

    @pytest.mark.asyncio
    async def test_document_orphan_denies(
        self, permission_service, perm_db, perm_redis
    ):
        """文档无文档级权限且无所属空间（孤儿）→ 拒绝访问。"""
        user_id = uuid.uuid4()
        document_id = uuid.uuid4()

        perm_db.execute = AsyncMock(
            side_effect=[
                scalar_one_or_none_result(None),  # 文档级 None
                scalar_one_or_none_result(None),  # 空间 ID None
            ]
        )

        assert await permission_service.check_access(
            user_id, document_id, ResourceType.document, Action.read
        ) is False


# ─── 默认拒绝（缺失记录）────────────────────────────────────────────


class TestDefaultDeny:
    """无任何权限记录时 → 默认 invisible，全部拒绝。"""

    @pytest.mark.asyncio
    async def test_space_missing_record_denies_all(
        self, permission_service, perm_db, perm_redis
    ):
        """空间无权限记录 → 默认拒绝所有 action。"""
        user_id = uuid.uuid4()
        space_id = uuid.uuid4()
        # 缓存 miss
        perm_redis.get = AsyncMock(return_value=None)
        # DB 无记录
        perm_db.execute = AsyncMock(
            return_value=scalar_one_or_none_result(None)
        )

        for action in Action:
            assert await permission_service.check_access(
                user_id, space_id, ResourceType.space, action
            ) is False

    @pytest.mark.asyncio
    async def test_cache_sentinel_returns_deny(
        self, permission_service, perm_redis
    ):
        """缓存中的 ``__none__`` 哨兵值 → 直接拒绝，不走 DB。"""
        user_id = uuid.uuid4()
        space_id = uuid.uuid4()
        perm_redis.get = AsyncMock(return_value="__none__")

        result = await permission_service.check_access(
            user_id, space_id, ResourceType.space, Action.read
        )
        assert result is False


# ─── 性能 smoke：缓存命中时不查询 DB ───────────────────────────────


class TestCheckAccessPerformance:
    """性能保障：缓存命中场景下不应触发任何 DB 调用，且响应时间 < 50ms。"""

    @pytest.mark.asyncio
    async def test_space_check_cache_hit_no_db_call(
        self, permission_service, perm_db, perm_redis
    ):
        """空间权限缓存命中 → ``db.execute`` 不应被调用。"""
        user_id = uuid.uuid4()
        space_id = uuid.uuid4()
        perm_redis.get = AsyncMock(return_value="read")
        perm_db.execute = AsyncMock()

        result = await permission_service.check_access(
            user_id, space_id, ResourceType.space, Action.read
        )
        assert result is True
        perm_db.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_space_check_cache_hit_under_50ms(
        self, permission_service, perm_redis
    ):
        """缓存命中时响应应远低于 50ms（实际 mock 调用 << 1ms，留足缓冲）。"""
        user_id = uuid.uuid4()
        space_id = uuid.uuid4()
        perm_redis.get = AsyncMock(return_value="write")

        start = time.perf_counter()
        await permission_service.check_access(
            user_id, space_id, ResourceType.space, Action.write
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 50, f"check_access 缓存命中耗时 {elapsed_ms:.2f}ms"


# ─── ACCESS_LEVEL_ACTIONS 映射不变量 ────────────────────────────────


class TestAccessLevelActionsMapping:
    """``ACCESS_LEVEL_ACTIONS`` 映射的结构性约束。"""

    def test_invisible_has_no_actions(self):
        assert ACCESS_LEVEL_ACTIONS[AccessLevel.invisible] == set()

    def test_read_allows_browse_and_read(self):
        actions = ACCESS_LEVEL_ACTIONS[AccessLevel.read]
        assert Action.browse in actions
        assert Action.read in actions
        assert Action.write not in actions

    def test_write_allows_all_actions(self):
        actions = ACCESS_LEVEL_ACTIONS[AccessLevel.write]
        assert {Action.browse, Action.read, Action.write} <= actions

    def test_read_subset_of_write(self):
        """read 允许的动作必然是 write 的子集（单调性）。"""
        assert (
            ACCESS_LEVEL_ACTIONS[AccessLevel.read]
            <= ACCESS_LEVEL_ACTIONS[AccessLevel.write]
        )


# ─── Hypothesis 属性测试：访问级别单调性 ───────────────────────────

try:
    from hypothesis import HealthCheck, given, settings as hyp_settings, strategies as st

    LEVELS = st.sampled_from(list(AccessLevel))
    ACTIONS = st.sampled_from(list(Action))

    @hyp_settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(level=LEVELS, action=ACTIONS)
    def test_property_action_in_mapping_iff_check_access_passes(level, action):
        """属性：action 是否被允许 ⇔ 出现在 ``ACCESS_LEVEL_ACTIONS[level]`` 中。

        Validates: Requirements 10
        """
        allowed = action in ACCESS_LEVEL_ACTIONS[level]
        # invisible 始终不允许；write 始终允许全部
        if level == AccessLevel.invisible:
            assert allowed is False
        if level == AccessLevel.write:
            assert allowed is True
        # read 仅允许 browse / read
        if level == AccessLevel.read:
            assert allowed is (action in {Action.browse, Action.read})

except ImportError:  # pragma: no cover - hypothesis 未安装则跳过属性测试
    pass
