"""注册接口与 ``AuthService.register`` 行为测试。

覆盖：
- 邮箱重复 → 409
- 密码长度不足 / 复杂度不足 → 422
- 成功注册 → 201 并返回用户信息

API 层通过 FastAPI TestClient + ``dependency_overrides`` 注入 mock 服务，
避免依赖真实 PostgreSQL / Redis。
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.auth import get_auth_service, router as auth_router
from app.core.exceptions import (
    ConflictException,
    ValidationException,
    register_exception_handlers,
)
from app.core.security import hash_password, validate_password_complexity
from app.services.auth_service import AuthService


# ─── Helpers / Fixtures ──────────────────────────────────────────────


def _make_app(auth_service: AuthService) -> FastAPI:
    """构建仅装载 auth router 的最小 FastAPI 应用。"""
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(auth_router)
    app.dependency_overrides[get_auth_service] = lambda: auth_service
    return app


@pytest.fixture
def stub_auth_service():
    """返回一个 AuthService 实例，但 db/redis 是 mock 的。"""
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    redis = AsyncMock()
    redis.hgetall = AsyncMock(return_value={})
    return AuthService(db=db, redis=redis)


# ─── Service 层 ──────────────────────────────────────────────────────


class TestRegisterService:
    """直接测试 AuthService.register。"""

    @pytest.mark.asyncio
    async def test_register_success(self, stub_auth_service):
        """新邮箱 + 合法密码 → 创建用户并 flush 入库。"""
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        stub_auth_service.db.execute = AsyncMock(return_value=result)

        user = await stub_auth_service.register(
            email="new@example.com",
            password="StrongPass1!",
            display_name="New User",
        )

        assert user.email == "new@example.com"
        assert user.password_hash is not None and user.password_hash != "StrongPass1!"
        stub_auth_service.db.add.assert_called_once()
        stub_auth_service.db.flush.assert_awaited()

    @pytest.mark.asyncio
    async def test_register_duplicate_email(self, stub_auth_service):
        """已存在邮箱 → ConflictException。"""
        existing = MagicMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = existing
        stub_auth_service.db.execute = AsyncMock(return_value=result)

        with pytest.raises(ConflictException, match="已被注册"):
            await stub_auth_service.register(
                email="dup@example.com",
                password="StrongPass1!",
                display_name="Dup",
            )
        stub_auth_service.db.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_register_password_too_short(self, stub_auth_service):
        """密码 <8 字符 → ValidationException。"""
        with pytest.raises(ValidationException, match="8"):
            await stub_auth_service.register(
                email="short@example.com",
                password="Ab1!",
                display_name="X",
            )

    @pytest.mark.asyncio
    async def test_register_password_low_complexity(self, stub_auth_service):
        """密码仅 2 类（小写+数字）→ ValidationException。"""
        with pytest.raises(ValidationException, match="三类"):
            await stub_auth_service.register(
                email="weak@example.com",
                password="abcdefgh12345",
                display_name="X",
            )


# ─── API 层 ──────────────────────────────────────────────────────────


class TestRegisterAPI:
    """通过 TestClient 测试 ``POST /api/auth/register``。"""

    def test_register_returns_201(self, stub_auth_service):
        # mock 行为：邮箱不存在，flush 后用户带 id
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        stub_auth_service.db.execute = AsyncMock(return_value=result)

        async def fake_flush() -> None:
            # 给 add 进来的对象赋一个 id
            mock_call = stub_auth_service.db.add.call_args
            if mock_call:
                user_obj = mock_call.args[0]
                user_obj.id = uuid.uuid4()

        stub_auth_service.db.flush = AsyncMock(side_effect=fake_flush)

        client = TestClient(_make_app(stub_auth_service))
        resp = client.post(
            "/api/auth/register",
            json={
                "email": "api@example.com",
                "password": "StrongPass1!",
                "display_name": "API User",
            },
        )
        assert resp.status_code == 201, resp.text
        data = resp.json()
        assert data["email"] == "api@example.com"
        assert data["display_name"] == "API User"
        assert "id" in data

    def test_register_duplicate_returns_409(self, stub_auth_service):
        existing = MagicMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = existing
        stub_auth_service.db.execute = AsyncMock(return_value=result)

        client = TestClient(_make_app(stub_auth_service))
        resp = client.post(
            "/api/auth/register",
            json={
                "email": "dup@example.com",
                "password": "StrongPass1!",
                "display_name": "Dup",
            },
        )
        assert resp.status_code == 409
        assert resp.json()["error"]["code"] == "Conflict"

    def test_register_short_password_returns_422(self, stub_auth_service):
        client = TestClient(_make_app(stub_auth_service))
        resp = client.post(
            "/api/auth/register",
            json={
                "email": "weak@example.com",
                "password": "Ab1!",  # Pydantic 的 min_length=8 校验先于业务校验
                "display_name": "Weak",
            },
        )
        assert resp.status_code == 422

    def test_register_low_complexity_returns_422(self, stub_auth_service):
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        stub_auth_service.db.execute = AsyncMock(return_value=result)

        client = TestClient(_make_app(stub_auth_service))
        resp = client.post(
            "/api/auth/register",
            json={
                "email": "weak@example.com",
                "password": "abcdefgh12345",  # 长度合法但仅 2 类
                "display_name": "Weak",
            },
        )
        assert resp.status_code == 422
        body = resp.json()
        assert body["error"]["code"] == "ValidationError"
        assert "三类" in body["error"]["message"]


# ─── PBT：密码复杂度函数属性测试 ─────────────────────────────────────


class TestPasswordComplexityProperty:
    """属性测试：``validate_password_complexity`` 与定义一致。

    Validates: Requirements 9.1
    """

    @staticmethod
    def _categories(s: str) -> int:
        cats = 0
        if any(c.isupper() for c in s):
            cats += 1
        if any(c.islower() for c in s):
            cats += 1
        if any(c.isdigit() for c in s):
            cats += 1
        if any(not c.isalnum() for c in s):
            cats += 1
        return cats

    def test_property_matches_specification(self):
        from hypothesis import given, settings as hyp_settings
        from hypothesis import strategies as st

        @hyp_settings(max_examples=200, deadline=None)
        @given(
            st.text(
                alphabet=st.characters(
                    min_codepoint=33, max_codepoint=126, blacklist_categories=("Cs",)
                ),
                min_size=1,
                max_size=80,
            )
        )
        def prop(password: str) -> None:
            ok, msg = validate_password_complexity(password)
            length_ok = 8 <= len(password) <= 64
            cats = self._categories(password)
            expected = length_ok and cats >= 3
            assert ok is expected, (password, ok, msg)

        prop()
