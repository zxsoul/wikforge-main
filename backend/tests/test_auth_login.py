"""登录接口与 JWT 签发行为测试。

覆盖：
- 邮箱不存在 → 401
- 密码错误 → 401（同一错误信息，不区分以避免枚举）
- 成功登录 → 返回 access/refresh，token claim 包含 sub/type/exp
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.auth import get_auth_service, router as auth_router
from app.core.exceptions import (
    UnauthorizedException,
    register_exception_handlers,
)
from app.core.security import decode_token, hash_password
from app.services.auth_service import AuthService


def _make_app(auth_service: AuthService) -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(auth_router)
    app.dependency_overrides[get_auth_service] = lambda: auth_service
    return app


@pytest.fixture
def stub_service():
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    redis = AsyncMock()
    redis.hgetall = AsyncMock(return_value={})
    redis.hset = AsyncMock()
    redis.expire = AsyncMock()
    redis.delete = AsyncMock()
    return AuthService(db=db, redis=redis)


def _existing_user(password: str) -> MagicMock:
    user = MagicMock()
    user.id = uuid.uuid4()
    user.email = "user@example.com"
    user.password_hash = hash_password(password)
    return user


# ─── Service 层 ──────────────────────────────────────────────────────


class TestLoginService:
    @pytest.mark.asyncio
    async def test_success_returns_token_pair(self, stub_service):
        user = _existing_user("StrongPass1!")
        result = MagicMock()
        result.scalar_one_or_none.return_value = user
        stub_service.db.execute = AsyncMock(return_value=result)

        token_pair = await stub_service.login("user@example.com", "StrongPass1!")
        assert set(token_pair) == {"access_token", "refresh_token", "token_type"}
        assert token_pair["token_type"] == "bearer"

        access_payload = decode_token(token_pair["access_token"])
        refresh_payload = decode_token(token_pair["refresh_token"])

        assert access_payload["sub"] == str(user.id)
        assert access_payload["type"] == "access"
        assert refresh_payload["type"] == "refresh"
        # 30 分钟有效期（容差 5s）
        delta = datetime.fromtimestamp(
            access_payload["exp"], tz=timezone.utc
        ) - datetime.now(timezone.utc)
        assert timedelta(minutes=29, seconds=55) < delta < timedelta(minutes=30, seconds=5)

        # 失败计数被清除
        stub_service.redis.delete.assert_awaited()

    @pytest.mark.asyncio
    async def test_wrong_password(self, stub_service):
        user = _existing_user("CorrectPass1!")
        result = MagicMock()
        result.scalar_one_or_none.return_value = user
        stub_service.db.execute = AsyncMock(return_value=result)

        with pytest.raises(UnauthorizedException, match="邮箱或密码错误"):
            await stub_service.login("user@example.com", "WrongPass1!")
        stub_service.redis.hset.assert_awaited()  # 失败被记录

    @pytest.mark.asyncio
    async def test_unknown_email(self, stub_service):
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        stub_service.db.execute = AsyncMock(return_value=result)

        with pytest.raises(UnauthorizedException, match="邮箱或密码错误"):
            await stub_service.login("unknown@example.com", "AnyPass1!")
        stub_service.redis.hset.assert_awaited()

    @pytest.mark.asyncio
    async def test_oidc_only_user_cannot_login_with_password(self, stub_service):
        """仅 OIDC 注册（password_hash 为空）的用户不能本地登录。"""
        user = MagicMock()
        user.id = uuid.uuid4()
        user.password_hash = None
        result = MagicMock()
        result.scalar_one_or_none.return_value = user
        stub_service.db.execute = AsyncMock(return_value=result)

        with pytest.raises(UnauthorizedException, match="邮箱或密码错误"):
            await stub_service.login("user@example.com", "AnyPass1!")


# ─── API 层 ──────────────────────────────────────────────────────────


class TestLoginAPI:
    def test_login_success(self, stub_service):
        user = _existing_user("StrongPass1!")
        result = MagicMock()
        result.scalar_one_or_none.return_value = user
        stub_service.db.execute = AsyncMock(return_value=result)

        client = TestClient(_make_app(stub_service))
        resp = client.post(
            "/api/auth/login",
            json={"email": "user@example.com", "password": "StrongPass1!"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["token_type"] == "bearer"
        assert decode_token(data["access_token"])["sub"] == str(user.id)

    def test_login_wrong_password_returns_401(self, stub_service):
        user = _existing_user("CorrectPass1!")
        result = MagicMock()
        result.scalar_one_or_none.return_value = user
        stub_service.db.execute = AsyncMock(return_value=result)

        client = TestClient(_make_app(stub_service))
        resp = client.post(
            "/api/auth/login",
            json={"email": "user@example.com", "password": "WrongPass1!"},
        )
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "Unauthorized"

    def test_login_unknown_email_returns_401(self, stub_service):
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        stub_service.db.execute = AsyncMock(return_value=result)

        client = TestClient(_make_app(stub_service))
        resp = client.post(
            "/api/auth/login",
            json={"email": "unknown@example.com", "password": "AnyPass1!"},
        )
        assert resp.status_code == 401
