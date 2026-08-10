"""Admin API for Document Profile management.

Provides:
- CRUD operations (POST/GET/PUT/DELETE /api/admin/profiles)
- Enable/disable toggle
- JSON import/export
- Version history
- Preview parsing with a specific profile
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.document_profile import DocumentProfile
from app.models.profile_version import ProfileVersion
from app.services.profile_candidate_service import (
    approve_candidate,
    get_candidate_metadata,
    list_candidates,
    reject_candidate,
    save_candidate,
)
from app.services.profile_version_service import (
    create_version_snapshot as _shared_create_version_snapshot,
)
from app.services.profile_version_service import (
    get_admin_user_id as _shared_get_admin_user_id,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/profiles", tags=["admin-profiles"])


# ─── Request/Response Schemas ──────────────────────────────────────────


class MatchRulesSchema(BaseModel):
    filename_regex: list[str] = Field(default_factory=list)
    content_regex: list[str] = Field(default_factory=list)
    min_content_match_count: int = 1


class HeadingRuleSchema(BaseModel):
    pattern: str
    level: int
    strip_pattern: bool = False


class BoilerplateSchema(BaseModel):
    detection_mode: str = "statistical"
    statistical_threshold: float = 0.5
    manual_patterns: list[str] = Field(default_factory=list)


class TableSchema(BaseModel):
    cross_page_merge: bool = True
    row_level_chunking: bool = False
    collapse_merged_cells: str = "describe"


class ChunkingSchema(BaseModel):
    min_tokens: int = 256
    max_tokens: int = 800
    overlap_tokens: int = 80
    respect_heading_level: int = 1
    protect_patterns: list[str] = Field(default_factory=list)


class ProfileCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None
    priority: int = 0
    enabled: bool = True
    match_rules: MatchRulesSchema = Field(default_factory=MatchRulesSchema)
    heading_rules: list[HeadingRuleSchema] = Field(default_factory=list)
    boilerplate: BoilerplateSchema = Field(default_factory=BoilerplateSchema)
    tables: TableSchema = Field(default_factory=TableSchema)
    chunking: ChunkingSchema = Field(default_factory=ChunkingSchema)
    domain_dictionary_id: str | None = None


class ProfileUpdateRequest(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=100)
    description: str | None = None
    priority: int | None = None
    enabled: bool | None = None
    match_rules: MatchRulesSchema | None = None
    heading_rules: list[HeadingRuleSchema] | None = None
    boilerplate: BoilerplateSchema | None = None
    tables: TableSchema | None = None
    chunking: ChunkingSchema | None = None
    domain_dictionary_id: str | None = None
    change_note: str | None = None


class ProfileResponse(BaseModel):
    id: str
    name: str
    description: str | None
    priority: int
    enabled: bool
    match_rules: dict
    heading_rules: list[dict]
    boilerplate: dict
    tables: dict
    chunking: dict
    domain_dictionary_id: str | None
    version: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ProfileListResponse(BaseModel):
    profiles: list[ProfileResponse]
    total: int


class ProfileVersionResponse(BaseModel):
    id: str
    profile_id: str
    version: int
    snapshot: dict
    changed_by: str
    change_note: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ProfileToggleRequest(BaseModel):
    enabled: bool


class ProfileImportRequest(BaseModel):
    profiles: list[dict]


class PreviewResponse(BaseModel):
    blocks: list[dict]
    features: dict
    matched_profile: str | None = None


class CandidateEvidenceSchema(BaseModel):
    """Universal Parser 推断候选 Profile 时的证据指标。

    与 ``UniversalParser.suggest_profile`` 返回的 ``metadata.evidence`` 结构
    完全对应，UI 用来在审核页展示「为什么我们推荐了这个 Profile」。
    """

    page_count: int = 0
    heading_count: int = 0
    table_count: int = 0
    boilerplate_candidates: int = 0
    avg_block_chars: float = 0.0


class CandidateMetadataSchema(BaseModel):
    """候选 envelope 的 metadata 块。"""

    status: str = Field(..., description="候选状态，必须是 'pending_approval'")
    source: str = Field(default="universal_parser")
    evidence: CandidateEvidenceSchema = Field(default_factory=CandidateEvidenceSchema)


class CandidateProfilePayload(BaseModel):
    """候选 Profile 的 inner profile dict（与 ProfileCreateRequest 同形态）。

    所有字段都允许默认值，因为 LLM 可能省略部分（例如 ``priority``）；服务层
    会负责把它们补齐成 ``DocumentProfile`` 的合法状态。
    """

    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None
    priority: int = 0
    enabled: bool = False
    match_rules: MatchRulesSchema = Field(default_factory=MatchRulesSchema)
    heading_rules: list[HeadingRuleSchema] = Field(default_factory=list)
    boilerplate: BoilerplateSchema = Field(default_factory=BoilerplateSchema)
    tables: TableSchema = Field(default_factory=TableSchema)
    chunking: ChunkingSchema = Field(default_factory=ChunkingSchema)
    domain_dictionary_id: str | None = None


class CandidateProfileRequest(BaseModel):
    """Universal Parser 推荐的候选 Profile 提交请求（任务 10.6）。

    输入是 ``UniversalParser.suggest_profile`` 返回的两层 envelope；这里只做
    Pydantic 浅校验，深入校验交给服务层。

    候选状态约定：
    - ``enabled=False`` 直至管理员审核通过
    - ``description`` 以 ``"[CANDIDATE] "`` 前缀标记，用于列表过滤与 UI 区分
    - 候选元数据（status / source / evidence）写入
      ``match_rules['__candidate__']`` sentinel
    """

    profile: CandidateProfilePayload
    metadata: CandidateMetadataSchema


class CandidateApprovalRequest(BaseModel):
    """``POST /candidates/{id}/approve`` 请求体。"""

    change_note: str | None = None
    priority: int | None = None
    enabled: bool = True


class CandidateProfileResponse(BaseModel):
    """候选列表的响应项 —— ``ProfileResponse`` + 解析回的 metadata envelope。"""

    profile: ProfileResponse
    metadata: CandidateMetadataSchema


class CandidateProfileListResponse(BaseModel):
    candidates: list[CandidateProfileResponse]
    total: int


# 候选 Profile 描述前缀（与 UI 协议一致）
CANDIDATE_DESCRIPTION_PREFIX = "[CANDIDATE] "


# ─── Helper Functions ──────────────────────────────────────────────────


def _profile_to_response(profile: DocumentProfile) -> ProfileResponse:
    """Convert ORM model to response schema."""
    return ProfileResponse(
        id=str(profile.id),
        name=profile.name,
        description=profile.description,
        priority=profile.priority,
        enabled=profile.enabled,
        match_rules=profile.match_rules or {},
        heading_rules=profile.heading_rules or [],
        boilerplate=profile.boilerplate or {},
        tables=profile.tables or {},
        chunking=profile.chunking or {},
        domain_dictionary_id=str(profile.domain_dictionary_id) if profile.domain_dictionary_id else None,
        version=profile.version,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )


async def _create_version_snapshot(
    db: AsyncSession,
    profile: DocumentProfile,
    changed_by: uuid.UUID,
    change_note: str | None = None,
) -> None:
    """Create a ProfileVersion snapshot for the current state of a profile.

    Thin wrapper kept for backwards compatibility — the real implementation
    lives in :mod:`app.services.profile_version_service` so candidate
    routes (任务 10.6) can reuse it without circular imports.
    """
    await _shared_create_version_snapshot(db, profile, changed_by, change_note)


# ─── CRUD Endpoints ───────────────────────────────────────────────────


@router.get("", response_model=ProfileListResponse)
async def list_profiles(
    enabled: bool | None = Query(None, description="Filter by enabled status"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> ProfileListResponse:
    """List all document profiles with optional filtering."""
    query = select(DocumentProfile)

    if enabled is not None:
        query = query.where(DocumentProfile.enabled == enabled)

    query = query.order_by(DocumentProfile.priority.desc(), DocumentProfile.updated_at.desc())

    # Count total
    count_query = select(DocumentProfile)
    if enabled is not None:
        count_query = count_query.where(DocumentProfile.enabled == enabled)
    count_result = await db.execute(count_query)
    total = len(count_result.scalars().all())

    # Paginate
    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    profiles = result.scalars().all()

    return ProfileListResponse(
        profiles=[_profile_to_response(p) for p in profiles],
        total=total,
    )


# ─── Candidate Profiles (任务 10.6) ────────────────────────────────────
# 这些路由必须放在 ``/{profile_id}`` 之前，否则会被通配的 ``/{profile_id}``
# 捕获 —— FastAPI 是按路由声明顺序匹配的。


def _candidate_to_response(profile: DocumentProfile) -> CandidateProfileResponse:
    """把 ORM Profile 折叠成候选响应：profile + 解析回的 metadata envelope。"""
    metadata = get_candidate_metadata(profile) or {
        "status": "pending_approval",
        "source": "universal_parser",
        "evidence": {},
    }
    return CandidateProfileResponse(
        profile=_profile_to_response(profile),
        metadata=CandidateMetadataSchema(
            status=metadata["status"],
            source=metadata["source"],
            evidence=CandidateEvidenceSchema(**metadata.get("evidence", {})),
        ),
    )


@router.post(
    "/candidates",
    response_model=CandidateProfileResponse,
    status_code=201,
)
async def create_candidate_profile(
    request: CandidateProfileRequest,
    db: AsyncSession = Depends(get_db),
) -> CandidateProfileResponse:
    """提交一份 Universal Parser 推荐的候选 Profile（任务 10.6）。

    请求体是 ``UniversalParser.suggest_profile`` 的两层 envelope；服务层会保证
    ``enabled=False`` 与 ``[CANDIDATE] `` 前缀，并在名字冲突时自动加 ``-N`` 后缀。
    """
    # 把 Pydantic 模型转回 envelope dict —— 服务层只接受 dict，避免与 schema 强耦合。
    candidate_dict = {
        "profile": request.profile.model_dump(),
        "metadata": request.metadata.model_dump(),
    }
    try:
        profile = await save_candidate(db, candidate_dict)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _candidate_to_response(profile)


@router.get("/candidates", response_model=CandidateProfileListResponse)
async def list_candidate_profiles(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> CandidateProfileListResponse:
    """列出所有「待管理员确认」的候选 Profile，按创建时间倒序。"""
    profiles, total = await list_candidates(db, skip=skip, limit=limit)
    return CandidateProfileListResponse(
        candidates=[_candidate_to_response(p) for p in profiles],
        total=total,
    )


@router.post(
    "/candidates/{profile_id}/approve",
    response_model=ProfileResponse,
)
async def approve_candidate_profile(
    profile_id: str,
    request: CandidateApprovalRequest,
    db: AsyncSession = Depends(get_db),
) -> ProfileResponse:
    """通过候选：去掉 candidate 标记，启用，自增版本，写入版本快照。"""
    try:
        profile = await approve_candidate(
            db,
            profile_id,
            change_note=request.change_note,
            enabled=request.enabled,
            priority_override=request.priority,
        )
    except ValueError as exc:
        message = str(exc)
        if "not found" in message:
            raise HTTPException(status_code=404, detail=message) from exc
        if "not a pending candidate" in message:
            raise HTTPException(status_code=404, detail=message) from exc
        if "collide" in message:
            raise HTTPException(status_code=409, detail=message) from exc
        raise HTTPException(status_code=400, detail=message) from exc
    return _profile_to_response(profile)


@router.post(
    "/candidates/{profile_id}/reject",
    status_code=204,
)
async def reject_candidate_profile(
    profile_id: str,
    db: AsyncSession = Depends(get_db),
) -> None:
    """拒绝候选：从数据库中删除该 Profile。"""
    try:
        await reject_candidate(db, profile_id)
    except ValueError as exc:
        message = str(exc)
        if "not found" in message or "not a pending candidate" in message:
            raise HTTPException(status_code=404, detail=message) from exc
        raise HTTPException(status_code=400, detail=message) from exc


# ─── /{profile_id} CRUD endpoints (must come AFTER /candidates) ──────


@router.get("/{profile_id}", response_model=ProfileResponse)
async def get_profile(
    profile_id: str,
    db: AsyncSession = Depends(get_db),
) -> ProfileResponse:
    """Get a single profile by ID."""
    result = await db.execute(
        select(DocumentProfile).where(DocumentProfile.id == uuid.UUID(profile_id))
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return _profile_to_response(profile)


@router.post("", response_model=ProfileResponse, status_code=201)
async def create_profile(
    request: ProfileCreateRequest,
    db: AsyncSession = Depends(get_db),
) -> ProfileResponse:
    """Create a new document profile."""
    # Check name uniqueness
    existing = await db.execute(
        select(DocumentProfile).where(DocumentProfile.name == request.name)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail=f"Profile with name '{request.name}' already exists")

    profile = DocumentProfile(
        name=request.name,
        description=request.description,
        priority=request.priority,
        enabled=request.enabled,
        match_rules=request.match_rules.model_dump(),
        heading_rules=[rule.model_dump() for rule in request.heading_rules],
        boilerplate=request.boilerplate.model_dump(),
        tables=request.tables.model_dump(),
        chunking=request.chunking.model_dump(),
        domain_dictionary_id=(
            uuid.UUID(request.domain_dictionary_id)
            if request.domain_dictionary_id
            else None
        ),
        version=1,
    )
    db.add(profile)
    await db.flush()
    await db.refresh(profile)

    # Create initial version snapshot (use a placeholder user for now)
    # In production, this would come from the authenticated user
    placeholder_user_id = await _get_admin_user_id(db)
    if placeholder_user_id:
        await _create_version_snapshot(db, profile, placeholder_user_id, "Initial creation")

    return _profile_to_response(profile)


@router.put("/{profile_id}", response_model=ProfileResponse)
async def update_profile(
    profile_id: str,
    request: ProfileUpdateRequest,
    db: AsyncSession = Depends(get_db),
) -> ProfileResponse:
    """Update an existing profile. Creates a version history entry."""
    result = await db.execute(
        select(DocumentProfile).where(DocumentProfile.id == uuid.UUID(profile_id))
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    # Check name uniqueness if name is being changed
    if request.name and request.name != profile.name:
        existing = await db.execute(
            select(DocumentProfile).where(
                DocumentProfile.name == request.name,
                DocumentProfile.id != uuid.UUID(profile_id),
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=409,
                detail=f"Profile with name '{request.name}' already exists",
            )

    # Apply updates
    if request.name is not None:
        profile.name = request.name
    if request.description is not None:
        profile.description = request.description
    if request.priority is not None:
        profile.priority = request.priority
    if request.enabled is not None:
        profile.enabled = request.enabled
    if request.match_rules is not None:
        profile.match_rules = request.match_rules.model_dump()
    if request.heading_rules is not None:
        profile.heading_rules = [rule.model_dump() for rule in request.heading_rules]
    if request.boilerplate is not None:
        profile.boilerplate = request.boilerplate.model_dump()
    if request.tables is not None:
        profile.tables = request.tables.model_dump()
    if request.chunking is not None:
        profile.chunking = request.chunking.model_dump()
    if request.domain_dictionary_id is not None:
        profile.domain_dictionary_id = (
            uuid.UUID(request.domain_dictionary_id)
            if request.domain_dictionary_id
            else None
        )

    # Increment version
    profile.version += 1

    await db.flush()
    await db.refresh(profile)

    # Create version snapshot
    placeholder_user_id = await _get_admin_user_id(db)
    if placeholder_user_id:
        await _create_version_snapshot(
            db, profile, placeholder_user_id, request.change_note
        )

    return _profile_to_response(profile)


@router.delete("/{profile_id}", status_code=204)
async def delete_profile(
    profile_id: str,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a profile. Cannot delete the default 'generic-text' profile."""
    result = await db.execute(
        select(DocumentProfile).where(DocumentProfile.id == uuid.UUID(profile_id))
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    if profile.name == "generic-text":
        raise HTTPException(
            status_code=400,
            detail="Cannot delete the default 'generic-text' profile",
        )

    await db.delete(profile)


# ─── Enable/Disable ───────────────────────────────────────────────────


@router.patch("/{profile_id}/toggle", response_model=ProfileResponse)
async def toggle_profile(
    profile_id: str,
    request: ProfileToggleRequest,
    db: AsyncSession = Depends(get_db),
) -> ProfileResponse:
    """Enable or disable a profile."""
    result = await db.execute(
        select(DocumentProfile).where(DocumentProfile.id == uuid.UUID(profile_id))
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    if profile.name == "generic-text" and not request.enabled:
        raise HTTPException(
            status_code=400,
            detail="Cannot disable the default 'generic-text' profile",
        )

    profile.enabled = request.enabled
    await db.flush()
    await db.refresh(profile)
    return _profile_to_response(profile)


# ─── Import/Export ─────────────────────────────────────────────────────


@router.get("/export/all")
async def export_profiles(
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Export all profiles as JSON."""
    result = await db.execute(
        select(DocumentProfile).order_by(DocumentProfile.priority.desc())
    )
    profiles = result.scalars().all()

    exported = []
    for p in profiles:
        exported.append({
            "name": p.name,
            "description": p.description,
            "priority": p.priority,
            "enabled": p.enabled,
            "match_rules": p.match_rules,
            "heading_rules": p.heading_rules,
            "boilerplate": p.boilerplate,
            "tables": p.tables,
            "chunking": p.chunking,
            "domain_dictionary_id": str(p.domain_dictionary_id) if p.domain_dictionary_id else None,
        })

    return {"profiles": exported, "exported_at": datetime.now(timezone.utc).isoformat(), "count": len(exported)}


@router.post("/import", response_model=dict)
async def import_profiles(
    request: ProfileImportRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Import profiles from JSON. Existing profiles with same name are updated."""
    imported = 0
    updated = 0
    errors: list[str] = []

    for profile_data in request.profiles:
        name = profile_data.get("name")
        if not name:
            errors.append("Profile missing 'name' field")
            continue

        try:
            # Check if profile exists
            result = await db.execute(
                select(DocumentProfile).where(DocumentProfile.name == name)
            )
            existing = result.scalar_one_or_none()

            if existing:
                # Update existing
                existing.description = profile_data.get("description", existing.description)
                existing.priority = profile_data.get("priority", existing.priority)
                existing.enabled = profile_data.get("enabled", existing.enabled)
                existing.match_rules = profile_data.get("match_rules", existing.match_rules)
                existing.heading_rules = profile_data.get("heading_rules", existing.heading_rules)
                existing.boilerplate = profile_data.get("boilerplate", existing.boilerplate)
                existing.tables = profile_data.get("tables", existing.tables)
                existing.chunking = profile_data.get("chunking", existing.chunking)
                existing.version += 1

                # Create version snapshot
                placeholder_user_id = await _get_admin_user_id(db)
                if placeholder_user_id:
                    await _create_version_snapshot(
                        db, existing, placeholder_user_id, "Imported (update)"
                    )
                updated += 1
            else:
                # Create new
                new_profile = DocumentProfile(
                    name=name,
                    description=profile_data.get("description", ""),
                    priority=profile_data.get("priority", 0),
                    enabled=profile_data.get("enabled", True),
                    match_rules=profile_data.get("match_rules", {}),
                    heading_rules=profile_data.get("heading_rules", []),
                    boilerplate=profile_data.get("boilerplate", {}),
                    tables=profile_data.get("tables", {}),
                    chunking=profile_data.get("chunking", {}),
                    version=1,
                )
                db.add(new_profile)
                imported += 1

        except Exception as e:
            errors.append(f"Error processing profile '{name}': {str(e)}")

    await db.flush()

    return {
        "imported": imported,
        "updated": updated,
        "errors": errors,
    }


# ─── Version History ───────────────────────────────────────────────────


@router.get("/{profile_id}/versions", response_model=list[ProfileVersionResponse])
async def get_profile_versions(
    profile_id: str,
    limit: int = Query(20, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
) -> list[ProfileVersionResponse]:
    """Get version history for a profile (most recent 20 versions)."""
    result = await db.execute(
        select(ProfileVersion)
        .where(ProfileVersion.profile_id == uuid.UUID(profile_id))
        .order_by(ProfileVersion.version.desc())
        .limit(limit)
    )
    versions = result.scalars().all()

    return [
        ProfileVersionResponse(
            id=str(v.id),
            profile_id=str(v.profile_id),
            version=v.version,
            snapshot=v.snapshot,
            changed_by=str(v.changed_by),
            change_note=v.change_note,
            created_at=v.created_at,
        )
        for v in versions
    ]


# ─── Preview ──────────────────────────────────────────────────────────


@router.post("/{profile_id}/preview", response_model=PreviewResponse)
async def preview_profile(
    profile_id: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> PreviewResponse:
    """Preview parsing results with a specific profile applied.

    Upload a sample document and see how the profile would process it.
    """
    import os
    import tempfile

    from app.services.parsers.registry import get_parser_registry
    from app.services.profile_matcher import ProfileMatcher, profile_from_dict

    # Get the profile
    result = await db.execute(
        select(DocumentProfile).where(DocumentProfile.id == uuid.UUID(profile_id))
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    # Save uploaded file to temp location
    ext = os.path.splitext(file.filename or "")[1] or ".txt"
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()

        # Parse the file
        registry = get_parser_registry()
        _ensure_default_parsers(registry)

        mime_type = file.content_type or ""
        try:
            parser = registry.select(tmp.name, mime_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"No parser available for file type '{ext}'",
            )

        parsed_doc = await parser.parse(tmp.name)

        # Extract features using ProfileMatcher
        profile_config = profile_from_dict({
            "id": str(profile.id),
            "name": profile.name,
            "description": profile.description,
            "priority": profile.priority,
            "enabled": profile.enabled,
            "match_rules": profile.match_rules,
            "heading_rules": profile.heading_rules,
            "boilerplate": profile.boilerplate,
            "tables": profile.tables,
            "chunking": profile.chunking,
            "domain_dictionary_id": str(profile.domain_dictionary_id) if profile.domain_dictionary_id else None,
            "version": profile.version,
            "created_at": profile.created_at,
            "updated_at": profile.updated_at,
        })

        matcher = ProfileMatcher(profiles=[profile_config])
        features = matcher.extract_features(parsed_doc, file.filename or "")

        # Return parsed blocks and features
        blocks = [
            {
                "type": block.type,
                "text": block.text,
                "page_number": block.page_number,
                "style": block.style,
            }
            for block in parsed_doc.blocks
        ]

        features_dict = {
            "filename": features.filename,
            "numbering_patterns": features.numbering_patterns,
            "header_footer_repetition": features.header_footer_repetition,
            "table_density": features.table_density,
            "page_count": features.page_count,
            "appears_scanned": features.appears_scanned,
            "avg_text_per_page": features.avg_text_per_page,
        }

        return PreviewResponse(
            blocks=blocks,
            features=features_dict,
            matched_profile=profile.name,
        )

    finally:
        os.unlink(tmp.name)


# ─── Internal Helpers ──────────────────────────────────────────────────


async def _get_admin_user_id(db: AsyncSession) -> uuid.UUID | None:
    """Get the admin user ID for version tracking.

    Backwards-compatible thin wrapper — delegates to
    :func:`app.services.profile_version_service.get_admin_user_id` so candidate
    routes can call the same logic without duplicating it.
    """
    return await _shared_get_admin_user_id(db)


def _ensure_default_parsers(registry) -> None:
    """Ensure default parsers are registered."""
    if registry.plugins:
        return

    from app.services.parsers.docx_parser import DocxParser
    from app.services.parsers.html_parser import HtmlParser
    from app.services.parsers.pdf_parser import PdfParser
    from app.services.parsers.pptx_parser import PptxParser
    from app.services.parsers.text_parser import TextParser

    for parser_class in [PdfParser, DocxParser, PptxParser, HtmlParser, TextParser]:
        try:
            registry.register(parser_class())
        except ValueError:
            pass
