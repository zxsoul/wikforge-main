"""安全工具：密码哈希、复杂度校验、JWT 签发与解析。

设计要点：
- **bcrypt 哈希**：直接使用 ``bcrypt`` 库（bcrypt 4.x 与 passlib 1.7 存在
  已知不兼容，passlib 在 backend 探测时会发送 73 字节探针触发 4.x 的 72
  字节硬限制；为保证 Python 3.13 + bcrypt 4.x 环境可运行测试，绕开 passlib）
- **密码复杂度**：与需求 9.1 对齐（8-64 字符、≥3 类）
- **JWT**：使用 ``python-jose``，sub/exp/type 三字段，type 区分 access/refresh
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from jose import JWTError, jwt

from app.core.config import get_settings

settings = get_settings()


# ─── Password Hashing ────────────────────────────────────────────────


# bcrypt 输入限制：≤72 字节（多余字符被忽略，实际编码后超过 72 会报错）
_BCRYPT_MAX_BYTES = 72


def _truncate_for_bcrypt(password: str) -> bytes:
    """按字节截断到 72 字节，避免 bcrypt 4.x 抛 ValueError。

    实际不会触达，因为 :func:`validate_password_complexity` 已经把密码限制
    在 64 字符以内（最坏情况下 4 字节字符 × 64 = 256 字节，所以这里仍需
    截断兜底）。
    """
    return password.encode("utf-8")[:_BCRYPT_MAX_BYTES]


def hash_password(password: str) -> str:
    """使用 bcrypt 哈希密码，返回字符串形式的 hash。"""
    hashed = bcrypt.hashpw(_truncate_for_bcrypt(password), bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证明文密码是否匹配 bcrypt hash。

    任意异常（如 hash 损坏）一律视为不匹配，避免暴露内部错误。
    """
    try:
        return bcrypt.checkpw(
            _truncate_for_bcrypt(plain_password),
            hashed_password.encode("utf-8"),
        )
    except (ValueError, TypeError):
        return False


# ─── Password Complexity Validation ──────────────────────────────────


def validate_password_complexity(password: str) -> tuple[bool, str]:
    """密码复杂度校验。

    规则（来自 requirements §9.1）：
    - 长度 8-64 字符
    - 大写字母 / 小写字母 / 数字 / 特殊字符 中至少包含 3 类

    Returns:
        (is_valid, error_message)。校验通过时 error_message 为空串。
    """
    if len(password) < 8:
        return False, "密码长度不能少于 8 个字符"
    if len(password) > 64:
        return False, "密码长度不能超过 64 个字符"

    categories = 0
    if re.search(r"[A-Z]", password):
        categories += 1
    if re.search(r"[a-z]", password):
        categories += 1
    if re.search(r"\d", password):
        categories += 1
    if re.search(r"[^A-Za-z0-9]", password):
        categories += 1

    if categories < 3:
        return False, "密码须包含大写字母、小写字母、数字和特殊字符中的至少三类"

    return True, ""


# ─── JWT Token ───────────────────────────────────────────────────────


def create_access_token(
    subject: str, extra_claims: dict[str, Any] | None = None
) -> str:
    """签发 Access Token，sub=用户 ID，type=access，默认 30 分钟。"""
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
    )
    to_encode: dict[str, Any] = {"sub": subject, "exp": expire, "type": "access"}
    if extra_claims:
        to_encode.update(extra_claims)
    return jwt.encode(
        to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
    )


def create_refresh_token(subject: str) -> str:
    """签发 Refresh Token，sub=用户 ID，type=refresh，默认 7 天。"""
    expire = datetime.now(timezone.utc) + timedelta(
        days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS
    )
    to_encode = {"sub": subject, "exp": expire, "type": "refresh"}
    return jwt.encode(
        to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
    )


def decode_token(token: str) -> dict[str, Any] | None:
    """解码并校验 JWT，无效或过期返回 ``None``。"""
    try:
        return jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
    except JWTError:
        return None
