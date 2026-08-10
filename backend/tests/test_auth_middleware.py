"""JWT 验证中间件 ``get_current_user`` 测试。

覆盖：
- 缺少 Authorization 头 → 401
- 错误的 scheme（如 Basic） → 401
- access token 无效 / 已过期 → 401
- 误用 refresh token 作为 access token → 401
- 合法 access token → 注入 User 实例
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jose import jwt

from app.api.auth import (
    UserResponse,
    get_auth_service,
    get_current_user,
    router as auth_router,
)
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.security import create_access_token, create_refresh_token
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


def _expired_access_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc) - timedelta(seconds=10),
        "type": "access",
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def _existing_user(user_id: uuid.UUID) -> MagicMock:
    user = MagicMock()
    user.id = user_id
    user.email = "user@example.com"
    user.display_name = "U"
    return user


class TestJWTMiddleware:
    def test_missing_header_returns_401(self, stub_service):
        client = TestClient(_make_app(stub_service))
        resp = client.get("/api/auth/me")
        assert resp.status_code == 401
        assert resp.json()["error"]["message"] == "缺少认证令牌"

    def test_wrong_scheme_returns_401(self, stub_service):
        client = TestClient(_make_app(stub_service))
        resp = client.get(
            "/api/auth/me", headers={"Authorization": "Basic abc"}
        )
        assert resp.status_code == 401

    def test_invalid_token_returns_401(self, stub_service):
        client = TestClient(_make_app(stub_service))
        resp = client.get(
            "/api/auth/me", headers={"Authorization": "Bearer invalid.token"}
        )
        assert resp.status_code == 401

    def test_expired_token_returns_401(self, stub_service):
        client = TestClient(_make_app(stub_service))
        token = _expired_access_token(str(uuid.uuid4()))
        resp = client.get(
            "/api/auth/me", headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 401
        assert "已过期" in resp.json()["error"]["message"]

    def test_refresh_token_rejected_as_access(self, stub_service):
        client = TestClient(_make_app(stub_service))
        token = create_refresh_token(subject=str(uuid.uuid4()))
        resp = client.get(
            "/api/auth/me", headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 401
        assert "Token 类型" in resp.json()["error"]["message"]

    def test_valid_token_returns_user(self, stub_service):
        user_id = uuid.uuid4()
        result = MagicMock()
        result.scalar_one_or_none.return_value = _existing_user(user_id)
        stub_service.db.execute = AsyncMock(return_value=result)

        client = TestClient(_make_app(stub_service))
        token = create_access_token(subject=str(user_id))
        resp = client.get(
            "/api/auth/me", headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == str(user_id)
        assert body["email"] == "user@example.com"

    def test_user_deleted_returns_401(self, stub_service):
        user_id = uuid.uuid4()
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        stub_service.db.execute = AsyncMock(return_value=result)

        client = TestClient(_make_app(stub_service))
        token = create_access_token(subject=str(user_id))
        resp = client.get(
            "/api/auth/me", headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 401
