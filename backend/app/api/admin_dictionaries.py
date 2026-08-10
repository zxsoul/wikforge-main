"""Admin API for Domain Dictionary management.

Provides:
- CRUD operations (POST/GET/PUT/DELETE /api/admin/dictionaries)
- Term management (add/remove terms)
- Synonym group management
- Import/export (CSV and JSON)
- Enable/disable toggle with IK sync
- Candidate term extraction
- Preset dictionary initialization

所有路由均通过 ``require_admin`` 强制管理员权限，与
``admin_profiles`` / ``admin_reviews`` 保持一致：未登录返回 401，登录但
非管理员返回 403。
"""

import logging
import uuid
from datetime import datetime
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import require_admin
from app.core.database import get_db
from app.models.domain_dictionary import DomainDictionary
from app.models.user import User
from app.services.dictionary_service import (
    DictionaryService,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/dictionaries", tags=["admin-dictionaries"])


# ─── Request/Response Schemas ──────────────────────────────────────────


class TermSchema(BaseModel):
    word: str = Field(..., min_length=1, max_length=30)
    pos: str | None = None
    weight: float = 1.0


class SynonymGroupSchema(BaseModel):
    primary: str = Field(..., min_length=1, max_length=30)
    synonyms: list[str] = Field(default_factory=list)


class DictionaryCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None
    terms: list[TermSchema] = Field(default_factory=list)
    synonyms: list[SynonymGroupSchema] = Field(default_factory=list)
    stop_words: list[str] = Field(default_factory=list)
    enabled: bool = True


class DictionaryUpdateRequest(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=100)
    description: str | None = None
    terms: list[TermSchema] | None = None
    synonyms: list[SynonymGroupSchema] | None = None
    stop_words: list[str] | None = None
    enabled: bool | None = None


class DictionaryResponse(BaseModel):
    id: str
    name: str
    description: str | None
    terms: list[dict]
    synonyms: list[dict]
    stop_words: list[str]
    enabled: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DictionaryListResponse(BaseModel):
    dictionaries: list[DictionaryResponse]
    total: int


class AddTermsRequest(BaseModel):
    terms: list[TermSchema]


class RemoveTermsRequest(BaseModel):
    words: list[str]


class AddSynonymGroupRequest(BaseModel):
    primary: str = Field(..., min_length=1, max_length=30)
    synonyms: list[str]


class RemoveSynonymGroupRequest(BaseModel):
    primary: str


class ToggleRequest(BaseModel):
    enabled: bool


class ImportJsonRequest(BaseModel):
    terms: list[dict] = Field(default_factory=list)
    synonyms: list[dict] = Field(default_factory=list)
    stop_words: list[str] = Field(default_factory=list)


class CandidateTermsRequest(BaseModel):
    documents_content: list[str]
    min_frequency: int = 3
    min_length: int = 2
    max_length: int = 10
    top_n: int = 50


class CandidateTermResponse(BaseModel):
    word: str
    frequency: int


# ─── Helper Functions ──────────────────────────────────────────────────


def _dictionary_to_response(d) -> DictionaryResponse:
    """Convert ORM model to response schema."""
    return DictionaryResponse(
        id=str(d.id),
        name=d.name,
        description=d.description,
        terms=d.terms or [],
        synonyms=d.synonyms or [],
        stop_words=d.stop_words or [],
        enabled=d.enabled,
        created_at=d.created_at,
        updated_at=d.updated_at,
    )


def _coerce_uuid(value: str, field_name: str = "dictionary_id") -> uuid.UUID:
    """Parse a UUID path/query param, returning 400 on malformed input.

    与 ``admin_reviews._coerce_uuid`` 同形态：让 ``f"/{not-a-uuid}"`` 这类
    请求落在 400（业务校验失败）而不是 500（``uuid.UUID`` 抛 ValueError 进
    DB 层）。
    """
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid {field_name}: {value!r}"
        ) from exc


async def _name_exists(
    db: AsyncSession,
    name: str,
    *,
    exclude_id: uuid.UUID | None = None,
) -> bool:
    """检查指定 name 是否已被其它 DomainDictionary 占用。

    POST/PUT 在写入之前先做这一次显式查询，让 409 路径不依赖底层异常字符串
    匹配，行为与 ``admin_profiles.create_profile`` 一致。
    """
    stmt = select(DomainDictionary).where(DomainDictionary.name == name)
    if exclude_id is not None:
        stmt = stmt.where(DomainDictionary.id != exclude_id)
    result = await db.execute(stmt)
    return result.scalar_one_or_none() is not None


# ─── CRUD Endpoints ───────────────────────────────────────────────────


@router.get("", response_model=DictionaryListResponse)
async def list_dictionaries(
    enabled: bool | None = Query(None, description="Filter by enabled status"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> DictionaryListResponse:
    """List all domain dictionaries with optional filtering."""
    service = DictionaryService(db)
    dictionaries, total = await service.list_dictionaries(
        enabled=enabled, skip=skip, limit=limit
    )
    return DictionaryListResponse(
        dictionaries=[_dictionary_to_response(d) for d in dictionaries],
        total=total,
    )


@router.get("/{dictionary_id}", response_model=DictionaryResponse)
async def get_dictionary(
    dictionary_id: str,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> DictionaryResponse:
    """Get a single dictionary by ID."""
    _coerce_uuid(dictionary_id)
    service = DictionaryService(db)
    dictionary = await service.get_dictionary(dictionary_id)
    if not dictionary:
        raise HTTPException(status_code=404, detail="Dictionary not found")
    return _dictionary_to_response(dictionary)


@router.post("", response_model=DictionaryResponse, status_code=201)
async def create_dictionary(
    request: DictionaryCreateRequest,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> DictionaryResponse:
    """Create a new domain dictionary。

    流程：
    1. 显式检查 ``name`` 唯一性（与 ``admin_profiles.create_profile`` 一致），
       命中返回 409，避免依赖底层 IntegrityError 字符串匹配。
    2. 委托 ``DictionaryService.create_dictionary`` 写入 + 触发 IK 同步。
    3. 术语校验失败由服务层抛 ``ValueError``，转为 422。
    """
    if await _name_exists(db, request.name):
        raise HTTPException(
            status_code=409,
            detail=f"Dictionary with name '{request.name}' already exists",
        )

    service = DictionaryService(db)
    try:
        dictionary = await service.create_dictionary(
            name=request.name,
            description=request.description,
            terms=[t.model_dump() for t in request.terms],
            synonyms=[s.model_dump() for s in request.synonyms],
            stop_words=request.stop_words,
            enabled=request.enabled,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return _dictionary_to_response(dictionary)


@router.put("/{dictionary_id}", response_model=DictionaryResponse)
async def update_dictionary(
    dictionary_id: str,
    request: DictionaryUpdateRequest,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> DictionaryResponse:
    """Update an existing dictionary."""
    dict_uuid = _coerce_uuid(dictionary_id)

    # name 改动时显式检查唯一性，命中返回 409
    if request.name is not None and await _name_exists(
        db, request.name, exclude_id=dict_uuid
    ):
        raise HTTPException(
            status_code=409,
            detail=f"Dictionary with name '{request.name}' already exists",
        )

    service = DictionaryService(db)
    try:
        dictionary = await service.update_dictionary(
            dictionary_id=dictionary_id,
            name=request.name,
            description=request.description,
            terms=[t.model_dump() for t in request.terms] if request.terms is not None else None,
            synonyms=[s.model_dump() for s in request.synonyms] if request.synonyms is not None else None,
            stop_words=request.stop_words,
            enabled=request.enabled,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not dictionary:
        raise HTTPException(status_code=404, detail="Dictionary not found")

    return _dictionary_to_response(dictionary)


@router.delete("/{dictionary_id}", status_code=204)
async def delete_dictionary(
    dictionary_id: str,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> None:
    """Delete a dictionary."""
    _coerce_uuid(dictionary_id)
    service = DictionaryService(db)
    deleted = await service.delete_dictionary(dictionary_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Dictionary not found")


# ─── Term Management ──────────────────────────────────────────────────


@router.post("/{dictionary_id}/terms", response_model=DictionaryResponse)
async def add_terms(
    dictionary_id: str,
    request: AddTermsRequest,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> DictionaryResponse:
    """Add terms to a dictionary."""
    _coerce_uuid(dictionary_id)
    service = DictionaryService(db)
    try:
        dictionary = await service.add_terms(
            dictionary_id=dictionary_id,
            new_terms=[t.model_dump() for t in request.terms],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not dictionary:
        raise HTTPException(status_code=404, detail="Dictionary not found")

    return _dictionary_to_response(dictionary)


@router.delete("/{dictionary_id}/terms", response_model=DictionaryResponse)
async def remove_terms(
    dictionary_id: str,
    request: RemoveTermsRequest,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> DictionaryResponse:
    """Remove terms from a dictionary by word."""
    _coerce_uuid(dictionary_id)
    service = DictionaryService(db)
    dictionary = await service.remove_terms(
        dictionary_id=dictionary_id,
        words=request.words,
    )
    if not dictionary:
        raise HTTPException(status_code=404, detail="Dictionary not found")

    return _dictionary_to_response(dictionary)


# ─── Synonym Management ───────────────────────────────────────────────


@router.post("/{dictionary_id}/synonyms", response_model=DictionaryResponse)
async def add_synonym_group(
    dictionary_id: str,
    request: AddSynonymGroupRequest,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> DictionaryResponse:
    """Add a synonym group to a dictionary."""
    _coerce_uuid(dictionary_id)
    service = DictionaryService(db)
    try:
        dictionary = await service.add_synonym_group(
            dictionary_id=dictionary_id,
            primary=request.primary,
            synonyms=request.synonyms,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not dictionary:
        raise HTTPException(status_code=404, detail="Dictionary not found")

    return _dictionary_to_response(dictionary)


@router.delete("/{dictionary_id}/synonyms", response_model=DictionaryResponse)
async def remove_synonym_group(
    dictionary_id: str,
    request: RemoveSynonymGroupRequest,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> DictionaryResponse:
    """Remove a synonym group by its primary term."""
    _coerce_uuid(dictionary_id)
    service = DictionaryService(db)
    dictionary = await service.remove_synonym_group(
        dictionary_id=dictionary_id,
        primary=request.primary,
    )
    if not dictionary:
        raise HTTPException(status_code=404, detail="Dictionary not found")

    return _dictionary_to_response(dictionary)


# ─── Enable/Disable ───────────────────────────────────────────────────


@router.patch("/{dictionary_id}/toggle", response_model=DictionaryResponse)
async def toggle_dictionary(
    dictionary_id: str,
    request: ToggleRequest,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> DictionaryResponse:
    """Enable or disable a dictionary. Disabled dictionaries are removed from IK."""
    _coerce_uuid(dictionary_id)
    service = DictionaryService(db)
    dictionary = await service.toggle_dictionary(
        dictionary_id=dictionary_id,
        enabled=request.enabled,
    )
    if not dictionary:
        raise HTTPException(status_code=404, detail="Dictionary not found")

    return _dictionary_to_response(dictionary)


# ─── Import/Export ─────────────────────────────────────────────────────


@router.get("/{dictionary_id}/export/json")
async def export_dictionary_json(
    dictionary_id: str,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> dict:
    """Export a dictionary as JSON."""
    _coerce_uuid(dictionary_id)
    service = DictionaryService(db)
    dictionary = await service.get_dictionary(dictionary_id)
    if not dictionary:
        raise HTTPException(status_code=404, detail="Dictionary not found")

    return service.export_as_json(dictionary)


@router.get("/{dictionary_id}/export/csv")
async def export_dictionary_csv(
    dictionary_id: str,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> PlainTextResponse:
    """Export dictionary terms as CSV (word,pos,weight)."""
    _coerce_uuid(dictionary_id)
    service = DictionaryService(db)
    dictionary = await service.get_dictionary(dictionary_id)
    if not dictionary:
        raise HTTPException(status_code=404, detail="Dictionary not found")

    csv_content = service.export_as_csv(dictionary)
    # ``Content-Disposition`` 必须是 ASCII，对中文 / 非 ASCII 文件名按
    # RFC 5987 用 ``filename*=UTF-8''<percent-encoded>`` 提供原始名，
    # ``filename="..."`` 留个 ASCII 兜底，避免 starlette 把整个 header 当
    # latin-1 编码时报 UnicodeEncodeError → 500。
    safe_name = (
        "".join(
            c if (c.isascii() and (c.isalnum() or c in "._-")) else "_"
            for c in dictionary.name
        )
        or "dictionary"
    )
    encoded_name = quote(dictionary.name, safe="")
    disposition = (
        f'attachment; filename="{safe_name}.csv"; '
        f"filename*=UTF-8''{encoded_name}.csv"
    )
    return PlainTextResponse(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": disposition},
    )


@router.post("/{dictionary_id}/import/json", response_model=DictionaryResponse)
async def import_dictionary_json(
    dictionary_id: str,
    request: ImportJsonRequest,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> DictionaryResponse:
    """Import terms, synonyms, and stop words from JSON into a dictionary."""
    _coerce_uuid(dictionary_id)
    service = DictionaryService(db)
    dictionary = await service.get_dictionary(dictionary_id)
    if not dictionary:
        raise HTTPException(status_code=404, detail="Dictionary not found")

    imported_data = service.import_from_json(request.model_dump())

    # Merge imported data with existing
    existing_terms = list(dictionary.terms or [])
    existing_words = {
        t.get("word") if isinstance(t, dict) else t for t in existing_terms
    }
    for term in imported_data["terms"]:
        word = term.get("word") if isinstance(term, dict) else term
        if word not in existing_words:
            existing_terms.append(term)
            existing_words.add(word)

    existing_synonyms = list(dictionary.synonyms or [])
    existing_primaries = {sg.get("primary") for sg in existing_synonyms}
    for sg in imported_data["synonyms"]:
        if sg.get("primary") not in existing_primaries:
            existing_synonyms.append(sg)
            existing_primaries.add(sg.get("primary"))

    existing_stop_words = list(dictionary.stop_words or [])
    stop_set = set(existing_stop_words)
    for sw in imported_data["stop_words"]:
        if sw not in stop_set:
            existing_stop_words.append(sw)
            stop_set.add(sw)

    try:
        updated = await service.update_dictionary(
            dictionary_id=dictionary_id,
            terms=existing_terms,
            synonyms=existing_synonyms,
            stop_words=existing_stop_words,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return _dictionary_to_response(updated)


@router.post("/{dictionary_id}/import/csv", response_model=DictionaryResponse)
async def import_dictionary_csv(
    dictionary_id: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> DictionaryResponse:
    """Import terms from a CSV file (word,pos,weight)."""
    _coerce_uuid(dictionary_id)
    service = DictionaryService(db)
    dictionary = await service.get_dictionary(dictionary_id)
    if not dictionary:
        raise HTTPException(status_code=404, detail="Dictionary not found")

    content = await file.read()
    try:
        csv_content = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="CSV file must be UTF-8 encoded",
        ) from exc
    imported_terms = service.import_from_csv(csv_content)

    # Merge with existing terms
    existing_terms = list(dictionary.terms or [])
    existing_words = {
        t.get("word") if isinstance(t, dict) else t for t in existing_terms
    }
    for term in imported_terms:
        if term.get("word") not in existing_words:
            existing_terms.append(term)
            existing_words.add(term.get("word"))

    try:
        updated = await service.update_dictionary(
            dictionary_id=dictionary_id,
            terms=existing_terms,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return _dictionary_to_response(updated)


# ─── Candidate Term Extraction ─────────────────────────────────────────


@router.post("/candidates/extract", response_model=list[CandidateTermResponse])
async def extract_candidates(
    request: CandidateTermsRequest,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> list[CandidateTermResponse]:
    """Extract candidate terms from document content.

    Analyzes document text for frequent terms not in existing dictionaries.
    """
    service = DictionaryService(db)
    candidates = await service.extract_candidate_terms(
        documents_content=request.documents_content,
        min_frequency=request.min_frequency,
        min_length=request.min_length,
        max_length=request.max_length,
        top_n=request.top_n,
    )
    return [
        CandidateTermResponse(word=c["word"], frequency=c["frequency"])
        for c in candidates
    ]


# ─── Preset Dictionary ─────────────────────────────────────────────────


@router.post("/preset/init", status_code=201)
async def init_preset_dictionaries(
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> dict:
    """Initialize preset dictionaries (e.g., Chinese stop words)."""
    from app.services.dictionary_service import ensure_preset_dictionaries

    await ensure_preset_dictionaries(db)
    return {"message": "Preset dictionaries initialized"}
