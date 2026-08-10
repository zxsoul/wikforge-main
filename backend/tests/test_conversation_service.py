"""ConversationService 单元测试（任务 16.5、需求 8.5）。

覆盖点：

- ``append`` / ``get_history`` / ``clear`` 的基本读写
- 角色合法性校验
- 20 轮（40 条）容量上限自动驱逐最旧消息
- TTL 写入为 1800 秒，且每次 ``append`` 都重置 TTL
- 不同 ``conversation_id`` 之间彼此隔离
- 历史中混入脏数据时容错
- ``conversation_id`` 为空字符串拒绝服务
"""

from __future__ import annotations

import json

import pytest
import pytest_asyncio

from app.services.conversation_service import (
    KEY_PREFIX,
    MAX_MESSAGES,
    MAX_TURNS,
    TTL_SECONDS,
    ConversationService,
)

# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def fake_redis_client():
    """基于 fakeredis 的真实异步 Redis 客户端。"""
    fakeredis = pytest.importorskip("fakeredis")
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()


@pytest_asyncio.fixture
async def service(fake_redis_client) -> ConversationService:
    """注入 fakeredis 的 ConversationService。"""
    return ConversationService(redis_client=fake_redis_client)


# ─── 基本读写 ──────────────────────────────────────────────────────────


class TestBasicAppendAndGet:
    """单条/多条消息追加与读取。"""

    @pytest.mark.asyncio
    async def test_append_and_get_single_turn(self, service):
        await service.append("conv-1", "user", "你好")
        await service.append("conv-1", "assistant", "你好，请问需要查询什么？")

        history = await service.get_history("conv-1")
        assert history == [
            {"role": "user", "content": "你好"},
            {
                "role": "assistant",
                "content": "你好，请问需要查询什么？",
            },
        ]

    @pytest.mark.asyncio
    async def test_get_history_for_unknown_conversation_returns_empty(
        self, service
    ):
        history = await service.get_history("never-existed")
        assert history == []

    @pytest.mark.asyncio
    async def test_history_preserves_insertion_order(self, service):
        for i in range(5):
            await service.append("conv", "user", f"q{i}")
            await service.append("conv", "assistant", f"a{i}")

        history = await service.get_history("conv")
        contents = [m["content"] for m in history]
        assert contents == [
            "q0", "a0", "q1", "a1", "q2", "a2", "q3", "a3", "q4", "a4",
        ]

    @pytest.mark.asyncio
    async def test_conversations_are_isolated(self, service):
        await service.append("c-A", "user", "A 的问题")
        await service.append("c-B", "user", "B 的问题")

        history_a = await service.get_history("c-A")
        history_b = await service.get_history("c-B")
        assert [m["content"] for m in history_a] == ["A 的问题"]
        assert [m["content"] for m in history_b] == ["B 的问题"]


# ─── 校验 ──────────────────────────────────────────────────────────────


class TestValidation:
    """非法输入应直接拒绝，不污染 Redis。"""

    @pytest.mark.asyncio
    async def test_rejects_invalid_role(self, service):
        with pytest.raises(ValueError):
            await service.append("c", "robot", "x")

    @pytest.mark.asyncio
    async def test_rejects_empty_conversation_id(self, service):
        with pytest.raises(ValueError):
            await service.append("", "user", "x")

    @pytest.mark.asyncio
    async def test_get_history_rejects_empty_conversation_id(self, service):
        with pytest.raises(ValueError):
            await service.get_history("")

    @pytest.mark.asyncio
    async def test_clear_rejects_empty_conversation_id(self, service):
        with pytest.raises(ValueError):
            await service.clear("")

    @pytest.mark.asyncio
    async def test_allows_empty_content(self, service):
        await service.append("c", "assistant", "")
        history = await service.get_history("c")
        assert history == [{"role": "assistant", "content": ""}]

    @pytest.mark.asyncio
    async def test_allows_unicode_content(self, service):
        text = "包含 emoji 😀 和换行\n第二行"
        await service.append("c", "user", text)
        history = await service.get_history("c")
        assert history[0]["content"] == text


# ─── 容量上限 ──────────────────────────────────────────────────────────


class TestCapacityLimit:
    """LTRIM 应保证最多保留最近 20 轮（40 条）消息。"""

    @pytest.mark.asyncio
    async def test_keeps_only_last_20_turns(self, service):
        # 写 25 轮 = 50 条消息
        for i in range(25):
            await service.append("c", "user", f"q{i}")
            await service.append("c", "assistant", f"a{i}")

        history = await service.get_history("c")
        assert len(history) == MAX_MESSAGES == 40
        # 最旧的应该是第 5 轮（i=5）的 user 消息
        assert history[0] == {"role": "user", "content": "q5"}
        # 最新的应该是第 24 轮（i=24）的 assistant 消息
        assert history[-1] == {"role": "assistant", "content": "a24"}

    @pytest.mark.asyncio
    async def test_keeps_exactly_20_turns_when_at_limit(self, service):
        for i in range(MAX_TURNS):
            await service.append("c", "user", f"q{i}")
            await service.append("c", "assistant", f"a{i}")

        history = await service.get_history("c")
        assert len(history) == MAX_MESSAGES
        assert history[0] == {"role": "user", "content": "q0"}

    @pytest.mark.asyncio
    async def test_does_not_truncate_below_limit(self, service):
        for i in range(3):
            await service.append("c", "user", f"q{i}")
        history = await service.get_history("c")
        assert len(history) == 3


# ─── TTL ───────────────────────────────────────────────────────────────


class TestTTL:
    """每次 append 都应把 TTL 重置为 1800 秒。"""

    @pytest.mark.asyncio
    async def test_first_append_sets_ttl_to_1800(self, service):
        await service.append("c", "user", "hello")
        ttl = await service.ttl("c")
        # fakeredis 的 ttl 精度为秒；允许 1798–1800
        assert 1790 <= ttl <= TTL_SECONDS

    @pytest.mark.asyncio
    async def test_subsequent_append_refreshes_ttl(
        self, service, fake_redis_client
    ):
        await service.append("c", "user", "first")
        # 手动把 TTL 缩到极小，模拟即将过期
        await fake_redis_client.expire(f"{KEY_PREFIX}c", 5)
        assert await service.ttl("c") <= 5

        await service.append("c", "assistant", "refresh me")
        ttl = await service.ttl("c")
        assert ttl > 1000, f"append 应当重置 TTL，但实际 ttl={ttl}"
        assert ttl <= TTL_SECONDS

    @pytest.mark.asyncio
    async def test_ttl_returns_minus2_for_unknown_conversation(self, service):
        # Redis TTL 协议：key 不存在返回 -2
        assert await service.ttl("never") == -2


# ─── clear ─────────────────────────────────────────────────────────────


class TestClear:
    @pytest.mark.asyncio
    async def test_clear_removes_history(self, service):
        await service.append("c", "user", "hi")
        await service.append("c", "assistant", "hello")
        await service.clear("c")
        assert await service.get_history("c") == []

    @pytest.mark.asyncio
    async def test_clear_unknown_conversation_silently_succeeds(self, service):
        await service.clear("never-existed")  # 不抛异常即可
        assert await service.get_history("never-existed") == []


# ─── 容错 ──────────────────────────────────────────────────────────────


class TestRobustness:
    """读取时遇到脏数据应静默跳过。"""

    @pytest.mark.asyncio
    async def test_skips_unparseable_entries(
        self, service, fake_redis_client
    ):
        key = f"{KEY_PREFIX}c"
        # 直接塞进一条非 JSON 的脏数据 + 一条合法消息
        await fake_redis_client.rpush(key, "not-json")
        await fake_redis_client.rpush(
            key, json.dumps({"role": "user", "content": "ok"})
        )

        history = await service.get_history("c")
        # 脏数据被忽略，合法消息保留
        assert history == [{"role": "user", "content": "ok"}]
