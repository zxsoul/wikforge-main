"""登录锁定行为测试（基于 fakeredis 的真实 Redis 行为）。

覆盖：
- 5 次失败后触发锁定（写入 ``locked_until``，设置 TTL）
- 锁定期间正确密码也被拒绝，错误信息含剩余分钟数
- 锁定过期后允许登录
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from app.core.exceptions import UnauthorizedException
from app.core.security import hash_password
from app.services.auth_service import (
    AuthService,
    LOCKOUT_DURATION_MINUTES,
    LOCKOUT_WINDOW_MINUTES,
    MAX_FAILED_ATTEMPTS,
)


# ─── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def known_user_password() -> str:
    return "StrongPass1!"


@pytest.fixture
def db_with_user(known_user_password):
    """返回一个 mock DB，scalar_one_or_none 永远返回带正确密码的用户。"""
    user = MagicMock()
    user.id = uuid.uuid4()
    user.email = "user@example.com"
    user.password_hash = hash_password(known_user_password)

    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = user
    db.execute = AsyncMock(return_value=result)
    return db, user


@pytest_asyncio.fixture
async def fake_redis_client():
    fakeredis = pytest.importorskip("fakeredis")
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()


# ─── Tests ───────────────────────────────────────────────────────────


class TestLockoutFlow:
    @pytest.mark.asyncio
    async def test_5_failures_trigger_lockout(self, db_with_user, fake_redis_client):
        db, user = db_with_user
        service = AuthService(db=db, redis=fake_redis_client)

        # 4 次错误密码 → 仍是凭证错误，未锁定
        for _ in range(MAX_FAILED_ATTEMPTS - 1):
            with pytest.raises(UnauthorizedException, match="邮箱或密码错误"):
                await service.login("user@example.com", "WrongPass1!")

        attempts_before = await fake_redis_client.hgetall(
            f"auth:lockout:user@example.com"
        )
        assert int(attempts_before["attempts"]) == MAX_FAILED_ATTEMPTS - 1
        assert "locked_until" not in attempts_before

        # 第 5 次错误 → 触发锁定
        with pytest.raises(UnauthorizedException, match="邮箱或密码错误"):
            await service.login("user@example.com", "WrongPass1!")

        data = await fake_redis_client.hgetall(f"auth:lockout:user@example.com")
        assert int(data["attempts"]) == MAX_FAILED_ATTEMPTS
        assert "locked_until" in data
        # TTL 应当为锁定时长（允许 5s 误差）
        ttl = await fake_redis_client.ttl(f"auth:lockout:user@example.com")
        assert LOCKOUT_DURATION_MINUTES * 60 - 5 <= ttl <= LOCKOUT_DURATION_MINUTES * 60

    @pytest.mark.asyncio
    async def test_correct_password_rejected_during_lock(
        self, db_with_user, fake_redis_client, known_user_password
    ):
        db, _ = db_with_user
        service = AuthService(db=db, redis=fake_redis_client)

        # 触发锁定
        for _ in range(MAX_FAILED_ATTEMPTS):
            with pytest.raises(UnauthorizedException):
                await service.login("user@example.com", "WrongPass1!")

        # 锁定期间用正确密码也应被拒绝，并提示剩余时间
        with pytest.raises(UnauthorizedException, match="锁定") as exc_info:
            await service.login("user@example.com", known_user_password)
        assert "分钟" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_login_after_lock_expires(
        self, db_with_user, fake_redis_client, known_user_password
    ):
        db, _ = db_with_user
        service = AuthService(db=db, redis=fake_redis_client)

        # 手动写入"已过期的锁定"
        expired = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        await fake_redis_client.hset(
            "auth:lockout:user@example.com",
            mapping={"attempts": "5", "locked_until": expired},
        )

        # 正确密码 → 成功登录
        token_pair = await service.login("user@example.com", known_user_password)
        assert "access_token" in token_pair

        # 登录成功后失败计数被清空
        assert await fake_redis_client.exists("auth:lockout:user@example.com") == 0

    @pytest.mark.asyncio
    async def test_window_ttl_set_on_each_failure(
        self, db_with_user, fake_redis_client
    ):
        db, _ = db_with_user
        service = AuthService(db=db, redis=fake_redis_client)
        with pytest.raises(UnauthorizedException):
            await service.login("user@example.com", "WrongPass1!")
        ttl = await fake_redis_client.ttl("auth:lockout:user@example.com")
        # 失败窗口 30 分钟（容差 5s）
        assert (
            LOCKOUT_WINDOW_MINUTES * 60 - 5 <= ttl <= LOCKOUT_WINDOW_MINUTES * 60
        )
