"""Database initialization script.

启用 PostgreSQL 扩展（``pgcrypto``、``pg_trgm``），创建初始管理员账号，
预置 3 个默认 Document Profile。

设计来源：
    .kiro/specs/enterprise-knowledge-base/design.md - 预置 Profile 章节、
        DocumentProfile 数据结构、Profile Matcher。

环境变量：
    INITIAL_ADMIN_EMAIL    初始管理员邮箱（默认 admin@wikforge.local）
    INITIAL_ADMIN_PASSWORD 初始管理员明文密码（默认 admin123!A）

用法::

    python -m app.scripts.init_db

脚本是幂等的：扩展使用 ``CREATE EXTENSION IF NOT EXISTS``；
管理员、Profile 通过 ``SELECT … WHERE name = …`` 检查后再插入。
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.models.document_profile import DocumentProfile
from app.models.user import User

logger = logging.getLogger(__name__)

# ─── 默认管理员账号（可通过环境变量覆盖）─────────────────────────────
DEFAULT_ADMIN_EMAIL = os.environ.get(
    "INITIAL_ADMIN_EMAIL", "admin@wikforge.local"
)
DEFAULT_ADMIN_PASSWORD = os.environ.get("INITIAL_ADMIN_PASSWORD", "admin123!A")
DEFAULT_ADMIN_DISPLAY_NAME = os.environ.get(
    "INITIAL_ADMIN_DISPLAY_NAME", "系统管理员"
)


# ─── 预置 Document Profiles ──────────────────────────────────────────
# 三个默认 Profile 对应 design.md 中的：
#   * generic-text         通用文本文档（默认兜底）
#   * chinese-technical-spec 中式技术规范（一/二/三、(一)/(二)、1/(1)/① 编号体系）
#   * scanned-pdf          扫描版 PDF（强制走 OCR + LLM 兜底）
DEFAULT_PROFILES: list[dict] = [
    {
        "name": "generic-text",
        "description": "通用文本文档 - 默认兜底 Profile，适用于大多数纯文本/简单格式文档",
        "priority": 0,
        "enabled": True,
        "match_rules": {
            "filename_regex": [],
            "content_regex": [],
            "min_content_match_count": 1,
        },
        "heading_rules": [
            {"pattern": r"^#{1,6}\s+", "level": 0, "strip_pattern": False},
        ],
        "boilerplate": {
            "detection_mode": "statistical",
            "statistical_threshold": 0.5,
            "manual_patterns": [],
        },
        "tables": {
            "cross_page_merge": True,
            "row_level_chunking": False,
            "collapse_merged_cells": "describe",
        },
        "chunking": {
            "min_tokens": 256,
            "max_tokens": 800,
            "overlap_tokens": 80,
            "respect_heading_level": 1,
            "protect_patterns": [],
        },
    },
    {
        "name": "chinese-technical-spec",
        "description": (
            "中式技术规范文档 - 适用于使用 一/二/三、(一)/(二)、1/2/3、"
            "(1)/(2)、①/② 编号体系的国标/行标/企标"
        ),
        "priority": 10,
        "enabled": True,
        "match_rules": {
            "filename_regex": [
                r".*规范.*",
                r".*标准.*",
                r".*规程.*",
                r".*技术.*要求.*",
                r"^GB[/\-T]?\d+",
                r"^DL[/\-T]?\d+",
            ],
            "content_regex": [
                r"^[一二三四五六七八九十]+[、．.]",
                r"^\([一二三四五六七八九十]+\)",
                r"^\d+\.\d+",
                r"^第[一二三四五六七八九十百]+[章节条款]",
            ],
            "min_content_match_count": 2,
        },
        "heading_rules": [
            {
                "pattern": r"^第[一二三四五六七八九十百]+[章]",
                "level": 1,
                "strip_pattern": False,
            },
            {
                "pattern": r"^第[一二三四五六七八九十百]+[节]",
                "level": 2,
                "strip_pattern": False,
            },
            {
                "pattern": r"^[一二三四五六七八九十]+[、．.]",
                "level": 2,
                "strip_pattern": False,
            },
            {
                "pattern": r"^\([一二三四五六七八九十]+\)",
                "level": 3,
                "strip_pattern": False,
            },
            {"pattern": r"^\d+[、．.]", "level": 3, "strip_pattern": False},
            {"pattern": r"^\(\d+\)", "level": 4, "strip_pattern": False},
            {
                "pattern": r"^[①②③④⑤⑥⑦⑧⑨⑩]",
                "level": 5,
                "strip_pattern": False,
            },
        ],
        "boilerplate": {
            "detection_mode": "both",
            "statistical_threshold": 0.5,
            "manual_patterns": [
                r"^第\s*\d+\s*页$",
                r"^\d+\s*/\s*\d+$",
                r"^(密级|版本|编号)[：:]",
                r".*\b(internal\s*use\s*only|机密)\b.*",
            ],
        },
        "tables": {
            "cross_page_merge": True,
            "row_level_chunking": True,
            "collapse_merged_cells": "describe",
        },
        "chunking": {
            "min_tokens": 256,
            "max_tokens": 800,
            "overlap_tokens": 80,
            "respect_heading_level": 2,
            "protect_patterns": [
                # 数值 + 单位（保护跨切片完整性）
                r"\d+(?:\.\d+)?\s*[a-zA-Z/%°℃℉μ]+",
                r"[±＋－]\s*\d+",
                r"\d+\s*[~～\-]\s*\d+",
                # 公式片段（最常见的赋值/比较结构）
                r"△\s*=\s*[\d.]+",
            ],
        },
    },
    {
        "name": "scanned-pdf",
        "description": (
            "扫描版 PDF 文档 - 文本层缺失或为图像，强制走 OCR + 多模态 LLM 兜底解析。"
        ),
        "priority": 5,
        "enabled": True,
        "match_rules": {
            "filename_regex": [r".*扫描.*", r".*scan.*"],
            "content_regex": [],
            "min_content_match_count": 1,
            # 通过特征探测命中：文本层字符数过低 / 文本层为空。
            "feature_rules": {
                "max_text_density_per_page": 0.05,
                "force_ocr": True,
            },
        },
        "heading_rules": [],
        "boilerplate": {
            "detection_mode": "statistical",
            "statistical_threshold": 0.5,
            "manual_patterns": [],
        },
        "tables": {
            "cross_page_merge": True,
            "row_level_chunking": False,
            "collapse_merged_cells": "describe",
        },
        "chunking": {
            "min_tokens": 256,
            "max_tokens": 800,
            "overlap_tokens": 80,
            "respect_heading_level": 1,
            "protect_patterns": [],
            # 扫描版需先经过 OCR/LLM 兜底，因此降低分块严格度。
            "force_universal_parser": True,
        },
    },
]


# ─── 步骤实现 ─────────────────────────────────────────────────────────


async def create_extensions(session: AsyncSession) -> None:
    """启用必需的 PostgreSQL 扩展：``pgcrypto``、``pg_trgm``。"""
    await session.execute(text('CREATE EXTENSION IF NOT EXISTS "pgcrypto"'))
    await session.execute(text('CREATE EXTENSION IF NOT EXISTS "pg_trgm"'))
    await session.commit()
    print("✓ PostgreSQL extensions ready (pgcrypto, pg_trgm)")


def _hash_password(plain: str) -> str:
    """Hash a plaintext password using bcrypt; fall back to passlib if needed."""
    try:
        from app.core.security import hash_password as _hp  # type: ignore

        return _hp(plain)
    except Exception:  # pragma: no cover - best-effort fallback
        try:
            from passlib.hash import bcrypt

            return bcrypt.hash(plain)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "无法对管理员密码进行哈希：未找到可用的 bcrypt 实现"
            ) from exc


async def create_admin_user(session: AsyncSession) -> uuid.UUID:
    """Create the initial admin account if it does not already exist."""
    result = await session.execute(
        select(User).where(User.email == DEFAULT_ADMIN_EMAIL)
    )
    existing = result.scalar_one_or_none()
    if existing:
        print(f"✓ Admin user already exists: {DEFAULT_ADMIN_EMAIL}")
        return existing.id

    admin = User(
        email=DEFAULT_ADMIN_EMAIL,
        password_hash=_hash_password(DEFAULT_ADMIN_PASSWORD),
        display_name=DEFAULT_ADMIN_DISPLAY_NAME,
    )
    session.add(admin)
    await session.commit()
    await session.refresh(admin)
    print(
        f"✓ Admin user created: {DEFAULT_ADMIN_EMAIL} "
        f"(请尽快通过界面修改默认密码)"
    )
    return admin.id


async def create_default_profiles(session: AsyncSession) -> None:
    """预置 3 个默认 DocumentProfile（已存在则跳过）。"""
    for profile_data in DEFAULT_PROFILES:
        result = await session.execute(
            select(DocumentProfile).where(
                DocumentProfile.name == profile_data["name"]
            )
        )
        existing = result.scalar_one_or_none()
        if existing:
            print(f"  • profile already exists: {profile_data['name']}")
            continue

        session.add(DocumentProfile(**profile_data))
        print(f"  + profile created: {profile_data['name']}")

    await session.commit()
    print("✓ Default Document Profiles ready")


async def init_database() -> None:
    """Run all initialization steps in a single async session."""
    print("=" * 60)
    print("Wikforge Database Initialization")
    print("=" * 60)

    async with AsyncSessionLocal() as session:
        await create_extensions(session)
        await create_admin_user(session)
        await create_default_profiles(session)

    print("=" * 60)
    print("Database initialization complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(init_database())
