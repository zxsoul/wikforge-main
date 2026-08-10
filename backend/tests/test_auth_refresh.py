"""Token 刷新接口与 ``AuthService.refresh_token`` 行为测试。

覆盖：
- 有效 refresh token → 返回新的 token 对
- 把 access token 当 refresh 用 → 401
- 过期 / 无效 token → 401
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jose import jwt

from app.api.auth import get_auth_service, router as auth_router
from app.core.config import get_settings
from app.core.exceptions import (
    UnauthorizedException,
    register_exception_handlers,
)
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
)
from app.services.auth_service import AuthService

settings = get_settings()


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
    redis = AsyncMock()
    redis.hgetall = AsyncMock(return_value={})
    return AuthService(db=db, redis=redis)


def _expired_refresh_token(user_id: str) -> str:
    """构造一个已过期的 refresh token（手动指定 exp）。"""
    payload = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc) - timedelta(minutes=1),
        "type": "refresh",
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


# ─── Service 层 ──────────────────────────────────────────────────────


class TestRefreshService:
    @pytest.mark.asyncio
    async def test_refresh_success(self, stub_service):
        user_id = uuid.uuid4()
        refresh = create_refresh_token(subject=str(user_id))

        existing = MagicMock()
        existing.id = user_id
        result = MagicMock()
        result.scalar_one_or_none.return_value = existing
        stub_service.db.execute = AsyncMock(return_value=result)

        new_pair = await stub_service.refresh_token(refresh)
        assert decode_token(new_pair["access_token"])["sub"] == str(user_id)
        assert decode_token(new_pair["refresh_token"])["type"] == "refresh"

    @pytest.mark.asyncio
    async def test_refresh_with_access_token_rejected(self, stub_service):
        user_id = str(uuid.uuid4())
        access = create_access_token(subject=user_id)
        with pytest.raises(UnauthorizedException, match="Token 类型"):
            await stub_service.refresh_token(access)

    @pytest.mark.asyncio
    async def test_refresh_invalid_token(self, stub_service):
        with pytest.raises(UnauthorizedException):
            await stub_service.refresh_token("garbage.token.value")

    @pytest.mark.asyncio
    async def test_refresh_expired_token(self, stub_service):
        with pytest.raises(UnauthorizedException, match="无效或已过期"):
            await stub_service.refresh_token(_expired_refresh_token(str(uuid.uuid4())))

    @pytest.mark.asyncio
    async def test_refresh_user_deleted(self, stub_service):
        user_id = str(uuid.uuid4())
        refresh = create_refresh_token(subject=user_id)
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        stub_service.db.execute = AsyncMock(return_value=result)

        with pytest.raises(UnauthorizedException, match="用户不存在"):
            await stub_service.refresh_token(refresh)


# ─── API 层 ──────────────────────────────────────────────────────────


class TestRefreshAPI:
    def test_refresh_success_returns_new_pair(self, stub_service):
        user_id = uuid.uuid4()
        existing = MagicMock()
        existing.id = user_id
        result = MagicMock()
        result.scalar_one_or_none.return_value = existing
        stub_service.db.execute = AsyncMock(return_value=result)

        client = TestClient(_make_app(stub_service))
        refresh = create_refresh_token(subject=str(user_id))
        resp = client.post("/api/auth/refresh", json={"refresh_token": refresh})
        assert resp.status_code == 200
        body = resp.json()
        assert decode_token(body["access_token"])["sub"] == str(user_id)

    def test_refresh_with_access_token_returns_401(self, stub_service):
        client = TestClient(_make_app(stub_service))
        access = create_access_token(subject=str(uuid.uuid4()))
        resp = client.post("/api/auth/refresh", json={"refresh_token": access})
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "Unauthorized"

    def test_refresh_expired_returns_401(self, stub_service):
        client = TestClient(_make_app(stub_service))
        resp = client.post(
            "/api/auth/refresh",
            json={"refresh_token": _expired_refresh_token(str(uuid.uuid4()))},
        )
        assert resp.status_code == 401
