"""认证服务：本地账号注册/登录、JWT 管理、登录锁定、OIDC 集成。

设计要点（见 design.md `Auth Service` 与 requirements §9）：

- **本地账号**：邮箱唯一、bcrypt 哈希、密码复杂度 8-64 字符且至少 3 类。
- **JWT**：Access Token 30 分钟、Refresh Token 7 天，sub=用户 ID，type 区分。
- **锁定**：30 分钟内连续 5 次失败 → 锁定 15 分钟，错误信息含剩余分钟数。
- **OIDC**：基于 ``authlib`` + ``httpx`` 实现 Discovery、authorize URL 生成、
  授权码换 token、userinfo 拉取以及自动绑定/创建用户。
- 业务异常通过 ``app.core.exceptions`` 统一抛出，由全局处理器返回标准信封。
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.exceptions import (
    ConflictException,
    UnauthorizedException,
    ValidationException,
)
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    validate_password_complexity,
    verify_password,
)
from app.models.user import User

settings = get_settings()

# 锁定常量（与需求 9.6 对齐）
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_WINDOW_MINUTES = 30
LOCKOUT_DURATION_MINUTES = 15


class AuthService:
    """认证业务核心。"""

    def __init__(self, db: AsyncSession, redis: Redis):
        self.db = db
        self.redis = redis

    # ──────────────────────── Registration ────────────────────────

    async def register(self, email: str, password: str, display_name: str) -> User:
        """注册本地账号。

        校验：
        - 密码复杂度（长度 8-64、≥3 类）
        - 邮箱唯一
        """
        is_valid, error_msg = validate_password_complexity(password)
        if not is_valid:
            raise ValidationException(error_msg)

        stmt = select(User).where(User.email == email)
        result = await self.db.execute(stmt)
        existing = result.scalar_one_or_none()
        if existing is not None:
            raise ConflictException("该邮箱已被注册")

        user = User(
            email=email,
            password_hash=hash_password(password),
            display_name=display_name,
        )
        self.db.add(user)
        await self.db.flush()
        return user

    # ──────────────────────── Login ────────────────────────

    async def login(self, email: str, password: str) -> dict[str, str]:
        """邮箱+密码登录。锁定计数对所有失败统一计入。"""
        await self._check_lockout(email)

        stmt = select(User).where(User.email == email)
        result = await self.db.execute(stmt)
        user = result.scalar_one_or_none()

        # 用户不存在或仅有 OIDC 身份（password_hash 为空）→ 视为凭证错误
        if user is None or not user.password_hash:
            await self._record_failed_attempt(email)
            raise UnauthorizedException("邮箱或密码错误")

        if not verify_password(password, user.password_hash):
            await self._record_failed_attempt(email)
            raise UnauthorizedException("邮箱或密码错误")

        await self._clear_failed_attempts(email)
        return self._issue_tokens(str(user.id))

    # ──────────────────────── Token Refresh ────────────────────────

    async def refresh_token(self, refresh_token: str) -> dict[str, str]:
        """用 Refresh Token 换发新的 token 对。"""
        payload = decode_token(refresh_token)
        if payload is None:
            raise UnauthorizedException("Refresh Token 无效或已过期，请重新认证")

        if payload.get("type") != "refresh":
            raise UnauthorizedException("无效的 Token 类型")

        subject = payload.get("sub")
        if not subject:
            raise UnauthorizedException("无效的 Token")

        stmt = select(User).where(User.id == subject)
        result = await self.db.execute(stmt)
        user = result.scalar_one_or_none()
        if user is None:
            raise UnauthorizedException("用户不存在")

        return self._issue_tokens(str(user.id))

    # ──────────────────────── Token Verification ────────────────────────

    async def verify_access_token(self, token: str) -> User:
        """验证 Access Token 并返回对应用户（中间件使用）。"""
        payload = decode_token(token)
        if payload is None:
            raise UnauthorizedException("Access Token 无效或已过期")

        if payload.get("type") != "access":
            raise UnauthorizedException("无效的 Token 类型")

        subject = payload.get("sub")
        if not subject:
            raise UnauthorizedException("无效的 Token")

        stmt = select(User).where(User.id == subject)
        result = await self.db.execute(stmt)
        user = result.scalar_one_or_none()
        if user is None:
            raise UnauthorizedException("用户不存在")

        return user

    # ──────────────────────── OIDC ────────────────────────

    @staticmethod
    def _ensure_oidc_configured() -> None:
        """OIDC 未配置时友好返回 422，避免 500。"""
        if not settings.OIDC_DISCOVERY_URL or not settings.OIDC_CLIENT_ID:
            raise ValidationException("OIDC 未配置，请联系管理员配置 OIDC 提供商")

    @staticmethod
    async def _fetch_oidc_metadata() -> dict[str, Any]:
        """通过 OIDC Discovery 端点拉取 provider 元数据。"""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(settings.OIDC_DISCOVERY_URL)
            resp.raise_for_status()
            return resp.json()

    @staticmethod
    def _provider_name() -> str:
        """从 Discovery URL 提取 provider 名（host 部分）。"""
        if not settings.OIDC_DISCOVERY_URL:
            return "oidc"
        return settings.OIDC_DISCOVERY_URL.split("//")[-1].split("/")[0]

    async def build_oidc_authorize_url(self) -> str:
        """生成 OIDC 授权跳转 URL。"""
        self._ensure_oidc_configured()
        from authlib.integrations.httpx_client import AsyncOAuth2Client

        metadata = await self._fetch_oidc_metadata()
        authorization_endpoint = metadata.get("authorization_endpoint")
        if not authorization_endpoint:
            raise ValidationException("OIDC 提供商未返回 authorization_endpoint")

        client = AsyncOAuth2Client(
            client_id=settings.OIDC_CLIENT_ID,
            client_secret=settings.OIDC_CLIENT_SECRET,
            redirect_uri=settings.OIDC_REDIRECT_URI,
            scope="openid email profile",
        )
        try:
            uri, _state = client.create_authorization_url(authorization_endpoint)
        finally:
            await client.aclose()
        return uri

    async def handle_oidc_callback(self, code: str) -> dict[str, str]:
        """OIDC 回调：换 token、拉 userinfo、自动创建/绑定用户、签发本地 token。"""
        self._ensure_oidc_configured()
        from authlib.integrations.httpx_client import AsyncOAuth2Client

        metadata = await self._fetch_oidc_metadata()
        token_endpoint = metadata.get("token_endpoint")
        userinfo_endpoint = metadata.get("userinfo_endpoint")
        if not token_endpoint or not userinfo_endpoint:
            raise ValidationException("OIDC 提供商缺少 token/userinfo endpoint")

        client = AsyncOAuth2Client(
            client_id=settings.OIDC_CLIENT_ID,
            client_secret=settings.OIDC_CLIENT_SECRET,
            redirect_uri=settings.OIDC_REDIRECT_URI,
        )
        try:
            await client.fetch_token(
                token_endpoint, code=code, grant_type="authorization_code"
            )
            resp = await client.get(userinfo_endpoint)
            userinfo = resp.json()
        finally:
            await client.aclose()

        subject = userinfo.get("sub", "")
        email = userinfo.get("email", "")
        display_name = (
            userinfo.get("name") or userinfo.get("preferred_username") or email
        )

        if not email or not subject:
            raise ValidationException("OIDC 提供商未返回必要的用户信息")

        provider = self._provider_name()
        user = await self.get_or_create_oidc_user(
            provider=provider,
            subject=subject,
            email=email,
            display_name=display_name,
        )
        return self._issue_tokens(str(user.id))

    async def get_or_create_oidc_user(
        self, provider: str, subject: str, email: str, display_name: str
    ) -> User:
        """根据 OIDC 身份查找或创建用户。

        匹配顺序：
        1. (provider, subject) 完全命中 → 返回该用户
        2. email 命中 → 绑定 OIDC 身份后返回
        3. 否则创建新用户
        """
        stmt = select(User).where(
            User.oidc_provider == provider, User.oidc_subject == subject
        )
        result = await self.db.execute(stmt)
        user = result.scalar_one_or_none()
        if user is not None:
            return user

        stmt = select(User).where(User.email == email)
        result = await self.db.execute(stmt)
        user = result.scalar_one_or_none()
        if user is not None:
            user.oidc_provider = provider
            user.oidc_subject = subject
            await self.db.flush()
            return user

        user = User(
            email=email,
            display_name=display_name,
            oidc_provider=provider,
            oidc_subject=subject,
        )
        self.db.add(user)
        await self.db.flush()
        return user

    # ──────────────────────── Lockout Mechanism ────────────────────────

    async def _check_lockout(self, email: str) -> None:
        """命中锁定窗口时抛出 401，并在错误信息中携带剩余分钟数。"""
        lock_key = self._lock_key(email)
        data = await self.redis.hgetall(lock_key)
        if not data:
            return

        locked_until_raw = data.get("locked_until")
        if not locked_until_raw:
            return

        try:
            locked_until_dt = datetime.fromisoformat(locked_until_raw)
        except ValueError:
            # 数据异常时兜底清理
            await self.redis.delete(lock_key)
            return

        now = datetime.now(timezone.utc)
        if now < locked_until_dt:
            remaining_seconds = (locked_until_dt - now).total_seconds()
            remaining_minutes = max(1, math.ceil(remaining_seconds / 60))
            raise UnauthorizedException(
                f"账号已被临时锁定，请在 {remaining_minutes} 分钟后重试"
            )
        # 锁定已过期 → 主动清理
        await self.redis.delete(lock_key)

    async def _record_failed_attempt(self, email: str) -> None:
        """记录一次失败，达到阈值即写入锁定时间。"""
        lock_key = self._lock_key(email)
        data = await self.redis.hgetall(lock_key)
        attempts = int(data.get("attempts", 0)) + 1

        if attempts >= MAX_FAILED_ATTEMPTS:
            locked_until = datetime.now(timezone.utc) + timedelta(
                minutes=LOCKOUT_DURATION_MINUTES
            )
            await self.redis.hset(
                lock_key,
                mapping={
                    "attempts": str(attempts),
                    "locked_until": locked_until.isoformat(),
                },
            )
            await self.redis.expire(lock_key, LOCKOUT_DURATION_MINUTES * 60)
        else:
            await self.redis.hset(lock_key, mapping={"attempts": str(attempts)})
            await self.redis.expire(lock_key, LOCKOUT_WINDOW_MINUTES * 60)

    async def _clear_failed_attempts(self, email: str) -> None:
        """登录成功 → 清空失败计数。"""
        await self.redis.delete(self._lock_key(email))

    @staticmethod
    def _lock_key(email: str) -> str:
        return f"auth:lockout:{email}"

    # ──────────────────────── Helpers ────────────────────────

    def _issue_tokens(self, user_id: str) -> dict[str, str]:
        """签发 access/refresh token 对。"""
        return {
            "access_token": create_access_token(subject=user_id),
            "refresh_token": create_refresh_token(subject=user_id),
            "token_type": "bearer",
        }
