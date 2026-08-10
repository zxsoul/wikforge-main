"""候选 Document Profile 持久化服务（任务 10.6）。

Universal Parser（任务 10.5）会基于 LLM 兜底解析的结果生成「候选 Profile」envelope：

```python
{
    "profile": <profile_to_dict-compatible dict>,
    "metadata": {
        "status": "pending_approval",
        "source": "universal_parser",
        "evidence": {...},
    },
}
```

本服务负责把候选保存到 ``DocumentProfile`` 表（不增加 schema 迁移），并提供
列表 / 通过 / 拒绝 三个生命周期方法。

约定：
- ``enabled = False`` —— 候选 Profile 不参与 ProfileMatcher 的真实匹配。
- ``description`` 以 :data:`CANDIDATE_DESCRIPTION_PREFIX` 开头，作为列表过滤
  的 SQL 索引友好谓词，同时给前端列表一个清晰可见的人类可读标识。
- 候选 envelope 中的 ``metadata`` 块（status / source / evidence）写入
  ``match_rules['__candidate__']`` —— ``profile_matcher.profile_from_dict``
  只读取 ``filename_regex`` / ``content_regex`` / ``min_content_match_count``
  这三个键，对未知键友好忽略，因此塞入 sentinel 不会污染 matcher 的行为。
- ``save_candidate`` 在 name 冲突时追加 ``-1`` / ``-2`` 后缀；``approve_candidate``
  会尝试还原原始名字，若还原后的名字已被其它非候选 Profile 占用则拒绝。

所有验证 / 业务异常通过 ``ValueError`` 抛出，路由层再翻译成 HTTP 4xx。
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document_profile import DocumentProfile
from app.services.profile_version_service import (
    create_version_snapshot,
    get_admin_user_id,
)

logger = logging.getLogger(__name__)


# 候选 description 前缀，与 ``app.api.admin_profiles.CANDIDATE_DESCRIPTION_PREFIX``
# 保持一致；这里再次定义是为了让服务层不依赖路由模块（避免循环导入）。
CANDIDATE_DESCRIPTION_PREFIX = "[CANDIDATE] "

# match_rules 中保存元数据的 sentinel key。
CANDIDATE_METADATA_KEY = "__candidate__"

# 候选 envelope 期望的 status 值。
EXPECTED_CANDIDATE_STATUS = "pending_approval"

# 候选 envelope 中默认的 source。
DEFAULT_CANDIDATE_SOURCE = "universal_parser"

# 唯一名生成时尝试的最大后缀数；超过即报名字冲突。10.6 不暴露到 API 层，但
# 防止 pathological case 一直循环。
_MAX_NAME_SUFFIX_ATTEMPTS = 1000

# 匹配 ``-<digits>`` 后缀，用于 approve 时尝试还原原始名字。
_NAME_SUFFIX_PATTERN = re.compile(r"^(?P<base>.+?)-(?P<n>\d+)$")


# ─── 私有工具 ────────────────────────────────────────────────────────


def _validate_envelope(candidate: Any) -> tuple[dict, dict]:
    """校验候选 envelope 的形状，返回 (profile_dict, metadata_dict)。

    Raises:
        ValueError: 任意字段缺失或类型不符。
    """
    if not isinstance(candidate, dict):
        raise ValueError("candidate envelope must be a dict")

    profile = candidate.get("profile")
    metadata = candidate.get("metadata")
    if not isinstance(profile, dict):
        raise ValueError("candidate envelope is missing 'profile' dict")
    if not isinstance(metadata, dict):
        raise ValueError("candidate envelope is missing 'metadata' dict")

    status = metadata.get("status")
    if status != EXPECTED_CANDIDATE_STATUS:
        raise ValueError(
            f"candidate metadata.status must be '{EXPECTED_CANDIDATE_STATUS}', got {status!r}"
        )

    name = profile.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("candidate profile is missing a non-empty 'name'")

    return profile, metadata


def _ensure_candidate_description(description: str | None) -> str:
    """保证 description 以 candidate 前缀开头。"""
    if description is None:
        description = ""
    if description.startswith(CANDIDATE_DESCRIPTION_PREFIX):
        return description
    # 把原描述拼到前缀后面，方便管理员审核时看到 LLM 推理的原始 description。
    return f"{CANDIDATE_DESCRIPTION_PREFIX}{description}".rstrip()


def _strip_candidate_description(description: str | None) -> str | None:
    """approve 时清理掉 candidate 前缀。"""
    if description is None:
        return None
    if description.startswith(CANDIDATE_DESCRIPTION_PREFIX):
        return description[len(CANDIDATE_DESCRIPTION_PREFIX):].lstrip() or None
    return description


def _attach_candidate_metadata(match_rules: dict | None, metadata: dict) -> dict:
    """把 envelope.metadata 写入 match_rules 的 sentinel 键，不破坏其它键。"""
    base = dict(match_rules or {})
    base[CANDIDATE_METADATA_KEY] = {
        "status": metadata.get("status", EXPECTED_CANDIDATE_STATUS),
        "source": metadata.get("source", DEFAULT_CANDIDATE_SOURCE),
        "evidence": metadata.get("evidence", {}),
    }
    return base


def _strip_candidate_metadata(match_rules: dict | None) -> dict:
    """approve 时把 sentinel 键去掉，并保证 match_rules 至少是空 dict。"""
    base = dict(match_rules or {})
    base.pop(CANDIDATE_METADATA_KEY, None)
    return base


def is_candidate_profile(profile: DocumentProfile) -> bool:
    """判断一个 ORM Profile 是否处于候选状态。

    判定条件：``description`` 以候选前缀开头，且 ``enabled`` 为 ``False``。
    两条同时成立才算候选 —— 单独看 description 可能误命中（管理员手动改名），
    单独看 enabled 也会和「停用 Profile」混淆。
    """
    if profile is None:
        return False
    desc = profile.description or ""
    return desc.startswith(CANDIDATE_DESCRIPTION_PREFIX) and not profile.enabled


def get_candidate_metadata(profile: DocumentProfile) -> dict | None:
    """从 ORM Profile 的 match_rules sentinel 中取回原始 envelope.metadata。

    Returns:
        ``{status, source, evidence}`` dict；如果不是候选 Profile 或没有 sentinel，
        返回 ``None``。
    """
    if not is_candidate_profile(profile):
        return None
    rules = profile.match_rules or {}
    meta = rules.get(CANDIDATE_METADATA_KEY)
    if not isinstance(meta, dict):
        return None
    # Defensive copy，避免调用方就地修改影响 ORM 的 JSONB 字段。
    return {
        "status": meta.get("status", EXPECTED_CANDIDATE_STATUS),
        "source": meta.get("source", DEFAULT_CANDIDATE_SOURCE),
        "evidence": dict(meta.get("evidence", {})),
    }


async def _resolve_unique_name(db: AsyncSession, base_name: str) -> str:
    """生成一个不与现有 Profile 冲突的名字。

    - 首先检查 base_name 是否可用。
    - 否则依次尝试 ``base_name-1`` / ``base_name-2`` ... 直到命中。
    """
    candidate = base_name
    suffix = 0
    while suffix <= _MAX_NAME_SUFFIX_ATTEMPTS:
        result = await db.execute(
            select(DocumentProfile).where(DocumentProfile.name == candidate)
        )
        if result.scalar_one_or_none() is None:
            return candidate
        suffix += 1
        candidate = f"{base_name}-{suffix}"
    raise ValueError(
        f"could not allocate a unique candidate profile name based on '{base_name}'"
    )


# ─── Public API ─────────────────────────────────────────────────────


async def save_candidate(db: AsyncSession, candidate: dict) -> DocumentProfile:
    """把 Universal Parser 产出的候选 envelope 持久化为「待审核」Profile。

    Args:
        db: AsyncSession。
        candidate: ``UniversalParser.suggest_profile`` 返回的两层 envelope。

    Returns:
        新建的 ``DocumentProfile`` ORM 实例（已 ``flush`` + ``refresh``，可直接
        序列化）。
    """
    profile_dict, metadata = _validate_envelope(candidate)

    base_name = profile_dict["name"].strip()
    name = await _resolve_unique_name(db, base_name)

    description = _ensure_candidate_description(profile_dict.get("description"))
    match_rules = _attach_candidate_metadata(profile_dict.get("match_rules"), metadata)

    profile = DocumentProfile(
        name=name,
        description=description,
        priority=int(profile_dict.get("priority", 0) or 0),
        # 候选必须为 disabled，避免被 ProfileMatcher 误用。
        enabled=False,
        match_rules=match_rules,
        heading_rules=profile_dict.get("heading_rules") or [],
        boilerplate=profile_dict.get("boilerplate") or {},
        tables=profile_dict.get("tables") or {},
        chunking=profile_dict.get("chunking") or {},
        domain_dictionary_id=(
            uuid.UUID(profile_dict["domain_dictionary_id"])
            if profile_dict.get("domain_dictionary_id")
            else None
        ),
        version=1,
    )
    db.add(profile)
    await db.flush()
    await db.refresh(profile)

    logger.info(
        "saved candidate profile id=%s name=%s evidence=%s",
        profile.id,
        profile.name,
        metadata.get("evidence"),
    )
    return profile


async def list_candidates(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 50,
) -> tuple[list[DocumentProfile], int]:
    """分页列出所有候选 Profile，按创建时间倒序。"""
    if skip < 0:
        skip = 0
    if limit <= 0:
        limit = 50

    base = select(DocumentProfile).where(
        DocumentProfile.enabled.is_(False),
        DocumentProfile.description.like(f"{CANDIDATE_DESCRIPTION_PREFIX}%"),
    )

    # total
    count_result = await db.execute(base)
    total = len(count_result.scalars().all())

    # paginated
    paginated = (
        base.order_by(DocumentProfile.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    page_result = await db.execute(paginated)
    profiles = list(page_result.scalars().all())

    return profiles, total


async def approve_candidate(
    db: AsyncSession,
    profile_id: uuid.UUID | str,
    change_note: str | None = None,
    *,
    enabled: bool = True,
    priority_override: int | None = None,
) -> DocumentProfile:
    """将候选 Profile 转正：清掉 sentinel + 前缀，自增版本，写入 ProfileVersion。

    Args:
        db: AsyncSession。
        profile_id: 候选 Profile 的 ID（UUID 或 str）。
        change_note: 写入版本快照的备注。
        enabled: 通过后是否启用，默认 ``True``。
        priority_override: 通过时一并调整优先级；``None`` 表示保持不变。

    Raises:
        ValueError: 找不到、不是候选、或者去掉 ``-N`` 后缀后名字与其他非候选
            Profile 冲突。

    Returns:
        更新后的 ORM Profile。
    """
    pid = profile_id if isinstance(profile_id, uuid.UUID) else uuid.UUID(str(profile_id))
    result = await db.execute(
        select(DocumentProfile).where(DocumentProfile.id == pid)
    )
    profile = result.scalar_one_or_none()
    if profile is None:
        raise ValueError(f"profile {pid} not found")
    if not is_candidate_profile(profile):
        raise ValueError(f"profile {pid} is not a pending candidate")

    # 如果 save_candidate 当时为了避免名字冲突追加了 ``-1`` / ``-2`` 后缀，
    # approve 时尝试还原到原始基础名 —— 这是一个体验优化，不强制成功。
    candidate_name = profile.name
    canonical_name = candidate_name
    suffix_match = _NAME_SUFFIX_PATTERN.match(candidate_name)
    if suffix_match:
        base = suffix_match.group("base")
        # 如果基础名空着的另一个 Profile 不存在，可以无损还原。
        existing_check = await db.execute(
            select(DocumentProfile).where(
                DocumentProfile.name == base,
                DocumentProfile.id != pid,
            )
        )
        clash = existing_check.scalar_one_or_none()
        if clash is None:
            canonical_name = base
        elif is_candidate_profile(clash):
            # 候选之间冲突属于历史遗留，保留 ``-N`` 后缀即可，不阻塞通过。
            canonical_name = candidate_name
        else:
            # 名称还原会撞到一个真实 Profile —— 拒绝通过，让管理员先处理。
            raise ValueError(
                f"approving candidate would collide with existing profile '{base}'"
            )

    # 即便没有 -N 后缀，仍然要求 canonical_name 与其它非候选 Profile 不同名。
    name_conflict_check = await db.execute(
        select(DocumentProfile).where(
            DocumentProfile.name == canonical_name,
            DocumentProfile.id != pid,
        )
    )
    other = name_conflict_check.scalar_one_or_none()
    if other is not None and not is_candidate_profile(other):
        raise ValueError(
            f"approving candidate would collide with existing profile '{canonical_name}'"
        )

    profile.name = canonical_name
    # 去掉前缀。如果剩余 description 为空字符串也归一化为 None，避免 UI 出现
    # 一个空白 description。
    profile.description = _strip_candidate_description(profile.description)
    # 去掉 match_rules 中的 sentinel。
    profile.match_rules = _strip_candidate_metadata(profile.match_rules)
    # 启用候选 Profile（默认）。
    profile.enabled = enabled
    if priority_override is not None:
        profile.priority = int(priority_override)
    profile.version = (profile.version or 1) + 1

    await db.flush()
    await db.refresh(profile)

    admin_id = await get_admin_user_id(db)
    if admin_id is not None:
        await create_version_snapshot(
            db,
            profile,
            admin_id,
            change_note or "Approved candidate profile",
        )

    logger.info("approved candidate profile id=%s name=%s", profile.id, profile.name)
    return profile


async def reject_candidate(
    db: AsyncSession,
    profile_id: uuid.UUID | str,
) -> None:
    """删除一个候选 Profile。

    Raises:
        ValueError: 不存在或不是候选状态。
    """
    pid = profile_id if isinstance(profile_id, uuid.UUID) else uuid.UUID(str(profile_id))
    result = await db.execute(
        select(DocumentProfile).where(DocumentProfile.id == pid)
    )
    profile = result.scalar_one_or_none()
    if profile is None:
        raise ValueError(f"profile {pid} not found")
    if not is_candidate_profile(profile):
        raise ValueError(f"profile {pid} is not a pending candidate")

    await db.delete(profile)
    logger.info("rejected (deleted) candidate profile id=%s name=%s", pid, profile.name)
