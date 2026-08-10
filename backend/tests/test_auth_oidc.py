"""OIDC 登录流程测试。

覆盖：
- 未配置 OIDC（``OIDC_DISCOVERY_URL`` 为空） → 422 友好错误
- ``get_or_create_oidc_user``：subject 命中 / email 命中绑定 / 创建新用户

OIDC 端到端跳转涉及外部 IdP，这里通过 mock authlib 与 Discovery 元数据的
方式验证业务行为，不真实发起 HTTP 请求。
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.auth import get_auth_service, router as auth_router
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.services.auth_service import AuthService

settings = get_settings()


# ─── App helpers ─────────────────────────────────────────────────────


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
    return AuthService(db=db, redis=redis)


# ─── 未配置 OIDC ─────────────────────────────────────────────────────


class TestOIDCNotConfigured:
    """未配置 OIDC 时应返回 422，而非 500。"""

    def test_authorize_returns_422(self, stub_service):
        # 默认 settings.OIDC_DISCOVERY_URL == ""（见 config.py）
        client = TestClient(_make_app(stub_service))
        resp = client.get("/api/auth/oidc/authorize", follow_redirects=False)
        assert resp.status_code == 422
        body = resp.json()
        assert body["error"]["code"] == "ValidationError"
        assert "OIDC" in body["error"]["message"]

    def test_callback_returns_422(self, stub_service):
        client = TestClient(_make_app(stub_service))
        resp = client.get("/api/auth/oidc/callback", params={"code": "abc"})
        assert resp.status_code == 422
        body = resp.json()
        assert body["error"]["code"] == "ValidationError"


# ─── get_or_create_oidc_user ────────────────────────────────────────


class TestGetOrCreateOIDCUser:
    @pytest.mark.asyncio
    async def test_existing_oidc_user_returned(self, stub_service):
        """OIDC (provider, subject) 命中 → 直接返回。"""
        existing = MagicMock()
        existing.email = "old@example.com"
        existing.oidc_provider = "keycloak.example.com"
        existing.oidc_subject = "sub-123"

        result = MagicMock()
        result.scalar_one_or_none.return_value = existing
        stub_service.db.execute = AsyncMock(return_value=result)

        user = await stub_service.get_or_create_oidc_user(
            provider="keycloak.example.com",
            subject="sub-123",
            email="ignored@example.com",
            display_name="X",
        )
        assert user is existing
        stub_service.db.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_bind_to_existing_email(self, stub_service):
        """provider/subject 未命中但邮箱已存在 → 绑定 OIDC。"""
        # 第一次查询（按 provider/subject）：无；第二次查询（按 email）：有
        not_found = MagicMock()
        not_found.scalar_one_or_none.return_value = None

        existing = MagicMock()
        existing.email = "user@example.com"
        existing.oidc_provider = None
        existing.oidc_subject = None

        found_email = MagicMock()
        found_email.scalar_one_or_none.return_value = existing

        stub_service.db.execute = AsyncMock(side_effect=[not_found, found_email])

        user = await stub_service.get_or_create_oidc_user(
            provider="okta.example.com",
            subject="okta-456",
            email="user@example.com",
            display_name="User",
        )
        assert user is existing
        assert user.oidc_provider == "okta.example.com"
        assert user.oidc_subject == "okta-456"
        stub_service.db.add.assert_not_called()
        stub_service.db.flush.assert_awaited()

    @pytest.mark.asyncio
    async def test_create_new_user(self, stub_service):
        """无任何匹配 → 创建新用户。"""
        not_found = MagicMock()
        not_found.scalar_one_or_none.return_value = None
        stub_service.db.execute = AsyncMock(side_effect=[not_found, not_found])

        user = await stub_service.get_or_create_oidc_user(
            provider="azure.example.com",
            subject="azure-789",
            email="brand-new@example.com",
            display_name="Brand New",
        )
        assert user.email == "brand-new@example.com"
        assert user.oidc_provider == "azure.example.com"
        assert user.oidc_subject == "azure-789"
        stub_service.db.add.assert_called_once()
        stub_service.db.flush.assert_awaited()


# ─── 端到端 callback：mock authlib 客户端 ────────────────────────────


class TestOIDCCallbackMocked:
    """通过 monkeypatch 让 authlib AsyncOAuth2Client 返回固定 userinfo。"""

    @pytest.mark.asyncio
    async def test_handle_oidc_callback_creates_user_and_issues_token(
        self, stub_service, monkeypatch
    ):
        # 1) 配置 OIDC
        monkeypatch.setattr(
            settings, "OIDC_DISCOVERY_URL",
            "https://keycloak.example.com/.well-known/openid-configuration",
            raising=False,
        )
        monkeypatch.setattr(settings, "OIDC_CLIENT_ID", "client-id", raising=False)
        monkeypatch.setattr(
            settings, "OIDC_CLIENT_SECRET", "client-secret", raising=False
        )
        monkeypatch.setattr(
            settings, "OIDC_REDIRECT_URI",
            "http://localhost/api/auth/oidc/callback", raising=False,
        )

        # 2) Mock 元数据拉取
        async def fake_fetch_metadata():
            return {
                "authorization_endpoint": "https://keycloak.example.com/auth",
                "token_endpoint": "https://keycloak.example.com/token",
                "userinfo_endpoint": "https://keycloak.example.com/userinfo",
            }

        monkeypatch.setattr(
            AuthService, "_fetch_oidc_metadata", staticmethod(fake_fetch_metadata)
        )

        # 3) Mock authlib 客户端
        mock_client = AsyncMock()
        mock_client.fetch_token = AsyncMock(return_value={"access_token": "x"})
        userinfo_resp = MagicMock()
        userinfo_resp.json = MagicMock(
            return_value={
                "sub": "sub-001",
                "email": "oidc@example.com",
                "name": "OIDC User",
            }
        )
        mock_client.get = AsyncMock(return_value=userinfo_resp)
        mock_client.aclose = AsyncMock()

        with patch(
            "authlib.integrations.httpx_client.AsyncOAuth2Client",
            return_value=mock_client,
        ):
            # 4) DB：未命中 OIDC、未命中邮箱 → 创建新用户
            not_found = MagicMock()
            not_found.scalar_one_or_none.return_value = None
            stub_service.db.execute = AsyncMock(side_effect=[not_found, not_found])

            token_pair = await stub_service.handle_oidc_callback("auth-code-xyz")

        assert "access_token" in token_pair
        assert "refresh_token" in token_pair
        stub_service.db.add.assert_called_once()
        # 创建的用户应携带 OIDC 信息
        created = stub_service.db.add.call_args.args[0]
        assert created.email == "oidc@example.com"
        assert created.oidc_subject == "sub-001"
        assert created.oidc_provider == "keycloak.example.com"

    @pytest.mark.asyncio
    async def test_callback_missing_email_rejected(self, stub_service, monkeypatch):
        monkeypatch.setattr(
            settings, "OIDC_DISCOVERY_URL",
            "https://keycloak.example.com/.well-known/openid-configuration",
            raising=False,
        )
        monkeypatch.setattr(settings, "OIDC_CLIENT_ID", "client-id", raising=False)

        async def fake_fetch_metadata():
            return {
                "authorization_endpoint": "https://keycloak.example.com/auth",
                "token_endpoint": "https://keycloak.example.com/token",
                "userinfo_endpoint": "https://keycloak.example.com/userinfo",
            }

        monkeypatch.setattr(
            AuthService, "_fetch_oidc_metadata", staticmethod(fake_fetch_metadata)
        )

        mock_client = AsyncMock()
        mock_client.fetch_token = AsyncMock(return_value={"access_token": "x"})
        userinfo_resp = MagicMock()
        userinfo_resp.json = MagicMock(return_value={"sub": "sub-001"})  # 无 email
        mock_client.get = AsyncMock(return_value=userinfo_resp)
        mock_client.aclose = AsyncMock()

        with patch(
            "authlib.integrations.httpx_client.AsyncOAuth2Client",
            return_value=mock_client,
        ):
            with pytest.raises(Exception) as exc_info:
                await stub_service.handle_oidc_callback("code")

        assert "OIDC" in str(exc_info.value) or "用户信息" in str(exc_info.value)
