"""检索反馈相关的 API 路由。

涵盖任务 17 反馈闭环：
- ``POST /api/feedback``：用户对一次检索/问答提交反馈（thumbs_up/thumbs_down/issue）
- ``GET /api/admin/feedback/list``：管理员查看反馈明细
- ``GET /api/admin/feedback/analysis``：反馈聚合分析
- ``GET /api/admin/feedback/patterns``：错误模式识别
- ``GET /api/admin/feedback/suggestions``：基于错误模式生成优化建议
- ``POST /api/admin/feedback/apply``：一键应用 Profile / 词典优化建议
- ``POST /api/admin/feedback/reprocess``：触发受影响文档批量重处理
- ``GET /api/admin/feedback/reprocess/{task_id}``：批量重处理进度

错误处理统一使用 :mod:`app.core.exceptions` 中的业务异常（``ValidationException``
等），由全局异常处理器映射为标准化响应信封。
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user, require_admin
from app.core.database import get_db
from app.core.exceptions import ValidationException
from app.models.user import User
from app.services.feedback_service import (
    ISSUE_CATEGORY_LABELS,
    FeedbackFilter,
    FeedbackService,
    IssueCategory,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["feedback"])


# ─── Request/Response Schemas ──────────────────────────────────────────


class FeedbackCreateRequest(BaseModel):
    """Request schema for creating feedback."""

    query: str = Field(..., min_length=1, max_length=2000, description="The search query")
    returned_results: list[str] = Field(
        default_factory=list, description="List of chunk IDs returned"
    )
    feedback_type: str = Field(
        ..., description="One of: thumbs_up, thumbs_down, issue"
    )
    issue_category: str | None = Field(
        None,
        description="Required when feedback_type is 'issue'. One of: irrelevant, missing_info, citation_error, format, other",
    )
    comment: str | None = Field(
        None, max_length=500, description="Optional text description"
    )
    related_profile_id: str | None = Field(
        None, description="Optional associated Document Profile ID"
    )


class FeedbackResponse(BaseModel):
    """Response schema for a single feedback entry."""

    id: str
    user_id: str
    query: str
    returned_results: list[str]
    feedback_type: str
    issue_category: str | None
    comment: str | None
    related_profile_id: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class FeedbackListResponse(BaseModel):
    """Response schema for feedback list."""

    feedbacks: list[FeedbackResponse]
    total: int


class AggregationResponse(BaseModel):
    """Response schema for feedback aggregation.

    各字段含义对应 :class:`FeedbackAggregation`：

    - ``by_issue_category``：``issue_category`` → 计数
    - ``by_profile``：``related_profile_id`` → 计数
    - ``by_document``：``returned_results`` 中的引用 ID → 命中次数
      （通常前端按 ``document_id`` / ``chunk_id`` 提交，由调用方约定）
    - ``by_date``：``YYYY-MM-DD`` → 当日反馈数量
    """

    total_count: int
    thumbs_up_count: int
    thumbs_down_count: int
    issue_count: int
    by_issue_category: dict[str, int]
    by_profile: dict[str, int]
    by_document: dict[str, int]
    by_date: dict[str, int]


class ErrorPatternResponse(BaseModel):
    """Response schema for a detected error pattern."""

    profile_id: str
    profile_name: str
    issue_category: str
    occurrence_count: int
    sample_queries: list[str]
    first_seen: datetime | None
    last_seen: datetime | None


class SuggestionResponse(BaseModel):
    """Response schema for an optimization suggestion."""

    type: str
    target_id: str
    target_name: str
    recommendation: dict
    evidence: list[str]
    confidence: float
    description: str


class ApplyProfileRequest(BaseModel):
    """Request schema for applying profile updates."""

    profile_id: str = Field(..., description="Profile ID to update")
    updates: dict = Field(
        ..., description="Fields to update (chunking, boilerplate, heading_rules)"
    )


class ApplyDictionaryRequest(BaseModel):
    """Request schema for applying dictionary updates."""

    dictionary_id: str = Field(..., description="Dictionary ID to update")
    new_terms: list[dict] = Field(
        ..., description="List of terms to add ({word, pos, weight})"
    )


class ApplyResponse(BaseModel):
    """Response schema for apply operations."""

    success: bool
    message: str
    reprocessing_task_id: str | None = None


class ReprocessRequest(BaseModel):
    """Request schema for triggering reprocessing."""

    profile_id: str | None = Field(
        None, description="Profile ID whose documents to reprocess"
    )
    dictionary_id: str | None = Field(
        None, description="Dictionary ID whose documents to reprocess"
    )
    document_ids: list[str] | None = Field(
        None, description="Explicit list of document IDs to reprocess"
    )


class ReprocessProgressResponse(BaseModel):
    """Response schema for reprocessing progress."""

    task_id: str
    total_documents: int
    processed_documents: int
    status: str
    progress_percent: float
    created_at: datetime
    error: str | None = None


class IssueCategoryItem(BaseModel):
    """单个问题类型选项（需求 9.2 / 18.2）。

    - ``value``：服务端枚举值，写入 ``SearchFeedback.issue_category``
    - ``label``：前端展示用的中文文案
    """

    value: str
    label: str


class IssueCategoriesResponse(BaseModel):
    """``GET /api/feedback/issue-categories`` 响应：返回全部可用问题类型。"""

    categories: list[IssueCategoryItem]


# ─── Helper Functions ──────────────────────────────────────────────────


def _feedback_to_response(feedback) -> FeedbackResponse:
    """Convert ORM model to response schema."""
    return FeedbackResponse(
        id=str(feedback.id),
        user_id=str(feedback.user_id),
        query=feedback.query,
        returned_results=feedback.returned_results or [],
        feedback_type=feedback.feedback_type,
        issue_category=feedback.issue_category,
        comment=feedback.comment,
        related_profile_id=(
            str(feedback.related_profile_id) if feedback.related_profile_id else None
        ),
        created_at=feedback.created_at,
    )


# ─── User Feedback Endpoint ───────────────────────────────────────────


@router.get(
    "/api/feedback/issue-categories",
    response_model=IssueCategoriesResponse,
)
async def list_issue_categories() -> IssueCategoriesResponse:
    """返回反馈问题类型的全部可选项（需求 9.2 / 18.2）。

    用于前端在用户标注「问题」反馈时动态渲染下拉框，避免硬编码枚举：

    - ``irrelevant``：结果不相关
    - ``missing_info``：缺少关键信息
    - ``citation_error``：引用错误
    - ``format``：格式问题
    - ``other``：其他

    枚举顺序与 :class:`app.services.feedback_service.IssueCategory` 一致，
    以便前端按声明顺序展示，不依赖 dict 插入顺序的潜在差异。
    """
    items = [
        IssueCategoryItem(
            value=category.value,
            label=ISSUE_CATEGORY_LABELS[category.value],
        )
        for category in IssueCategory
    ]
    return IssueCategoriesResponse(categories=items)


@router.post("/api/feedback", response_model=FeedbackResponse, status_code=201)
async def create_feedback(
    request: FeedbackCreateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> FeedbackResponse:
    """提交检索质量反馈（需求 9.1 / 18.1）。

    支持三种反馈类型：

    - ``thumbs_up``：结果有帮助
    - ``thumbs_down``：结果无帮助
    - ``issue``：标注具体问题，需要同时提供 ``issue_category``

    可选 ``issue_category``：

    - ``irrelevant``：结果与查询不相关
    - ``missing_info``：缺少关键信息
    - ``citation_error``：引用错误
    - ``format``：格式问题
    - ``other``：其他

    鉴权：依赖 :func:`app.api.auth.get_current_user`，未登录返回 401。
    """
    service = FeedbackService(db)

    try:
        feedback = await service.create_feedback(
            user_id=current_user.id,
            query=request.query,
            returned_results=request.returned_results,
            feedback_type=request.feedback_type,
            issue_category=request.issue_category,
            comment=request.comment,
            related_profile_id=request.related_profile_id,
        )
    except ValueError as e:
        # FeedbackService 抛出的字段级业务校验失败统一映射为 422
        raise ValidationException(str(e)) from e

    return _feedback_to_response(feedback)


# ─── Admin Analysis Endpoints ─────────────────────────────────────────


@router.get("/api/admin/feedback/list", response_model=FeedbackListResponse)
async def list_feedbacks(
    profile_id: str | None = Query(None, description="Filter by profile ID"),
    document_id: str | None = Query(
        None,
        description=(
            "按文档 / chunk 标识过滤，命中条件为 returned_results 数组中包含该 ID"
        ),
    ),
    feedback_type: str | None = Query(None, description="Filter by feedback type"),
    issue_category: str | None = Query(None, description="Filter by issue category"),
    start_date: str | None = Query(None, description="Start date (ISO format)"),
    end_date: str | None = Query(None, description="End date (ISO format)"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> FeedbackListResponse:
    """List feedback entries with optional filtering.

    通过 ``require_admin`` 强制管理员权限：未登录返回 401，登录但非管理员
    返回 403（详见 :func:`app.api.auth.require_admin`）。
    """
    filter = FeedbackFilter(
        profile_id=profile_id,
        document_id=document_id,
        feedback_type=feedback_type,
        issue_category=issue_category,
        start_date=(
            datetime.fromisoformat(start_date) if start_date else None
        ),
        end_date=datetime.fromisoformat(end_date) if end_date else None,
    )

    service = FeedbackService(db)
    feedbacks, total = await service.list_feedbacks(filter=filter, skip=skip, limit=limit)

    return FeedbackListResponse(
        feedbacks=[_feedback_to_response(fb) for fb in feedbacks],
        total=total,
    )


@router.get("/api/admin/feedback/analysis", response_model=AggregationResponse)
async def get_feedback_analysis(
    profile_id: str | None = Query(None, description="Filter by profile ID"),
    document_id: str | None = Query(
        None,
        description=(
            "按文档 / chunk 标识过滤，仅聚合 returned_results 包含该 ID 的反馈"
        ),
    ),
    feedback_type: str | None = Query(None, description="Filter by feedback type"),
    issue_category: str | None = Query(None, description="Filter by issue category"),
    start_date: str | None = Query(None, description="Start date (ISO format)"),
    end_date: str | None = Query(None, description="End date (ISO format)"),
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> AggregationResponse:
    """反馈聚合分析（任务 17.4 / 需求 9.4）。

    支持按 Profile、文档（基于 ``returned_results``）、查询类型
    （``feedback_type`` + ``issue_category``）、时间范围（``start_date`` /
    ``end_date``）多维过滤与聚合，响应字段同时给出：

    - 总量及三类反馈分项计数
    - ``by_issue_category`` / ``by_profile`` / ``by_document`` / ``by_date``

    通过 ``require_admin`` 强制管理员权限：未登录返回 401，登录但非管理员
    返回 403。
    """
    filter = FeedbackFilter(
        profile_id=profile_id,
        document_id=document_id,
        feedback_type=feedback_type,
        issue_category=issue_category,
        start_date=(
            datetime.fromisoformat(start_date) if start_date else None
        ),
        end_date=datetime.fromisoformat(end_date) if end_date else None,
    )

    service = FeedbackService(db)
    aggregation = await service.aggregate_feedback(filter=filter)

    return AggregationResponse(
        total_count=aggregation.total_count,
        thumbs_up_count=aggregation.thumbs_up_count,
        thumbs_down_count=aggregation.thumbs_down_count,
        issue_count=aggregation.issue_count,
        by_issue_category=aggregation.by_issue_category,
        by_profile=aggregation.by_profile,
        by_document=aggregation.by_document,
        by_date=aggregation.by_date,
    )


@router.get(
    "/api/admin/feedback/patterns", response_model=list[ErrorPatternResponse]
)
async def get_error_patterns(
    min_occurrences: int = Query(3, ge=1, description="Minimum occurrences to flag"),
    days_lookback: int = Query(30, ge=1, le=365, description="Days to look back"),
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> list[ErrorPatternResponse]:
    """错误模式检测（任务 17.5 / 需求 9.5）。

    识别同一 Profile 下相同 ``issue_category`` 重复出现 ≥ ``min_occurrences``
    次（默认 3 次）的反馈聚合，作为后续优化建议生成的输入。

    - ``min_occurrences``：触发阈值（默认 3，对应需求 9.5「重复出现 N 次」）。
    - ``days_lookback``：仅统计最近 N 天的反馈，避免历史噪声干扰新近模式。

    通过 ``require_admin`` 强制管理员权限：未登录返回 401，登录但非管理员
    返回 403。
    """
    service = FeedbackService(db)
    patterns = await service.detect_error_patterns(
        min_occurrences=min_occurrences,
        days_lookback=days_lookback,
    )

    return [
        ErrorPatternResponse(
            profile_id=p.profile_id,
            profile_name=p.profile_name,
            issue_category=p.issue_category,
            occurrence_count=p.occurrence_count,
            sample_queries=p.sample_queries,
            first_seen=p.first_seen,
            last_seen=p.last_seen,
        )
        for p in patterns
    ]


@router.get(
    "/api/admin/feedback/suggestions", response_model=list[SuggestionResponse]
)
async def get_optimization_suggestions(
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> list[SuggestionResponse]:
    """基于错误模式生成优化建议（任务 17.6 / 需求 9.6）。

    内部委托给 :meth:`FeedbackService.generate_suggestions`，按
    ``issue_category`` 映射到不同的 ``SuggestionType``：

    - ``irrelevant`` / ``missing_info`` → ``adjust_chunking``
    - ``citation_error`` → ``update_boilerplate``
    - ``format`` → ``adjust_heading_rules``
    - ``general_negative``（无明确问题类型的 thumbs_down）→ ``add_term``

    每条建议附带 ``confidence``（依据 ``occurrence_count`` 估算并上限 1.0）、
    ``recommendation``（具体调整动作）、``evidence``（样本查询）以及面向管
    理员的中文 ``description``。

    通过 ``require_admin`` 强制管理员权限：未登录返回 401，登录但非管理员
    返回 403。
    """
    service = FeedbackService(db)
    suggestions = await service.generate_suggestions()

    return [
        SuggestionResponse(
            type=s.type,
            target_id=s.target_id,
            target_name=s.target_name,
            recommendation=s.recommendation,
            evidence=s.evidence,
            confidence=s.confidence,
            description=s.description,
        )
        for s in suggestions
    ]


# ─── Apply Suggestions Endpoints ──────────────────────────────────────


@router.post("/api/admin/feedback/apply/profile", response_model=ApplyResponse)
async def apply_profile_suggestion(
    request: ApplyProfileRequest,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> ApplyResponse:
    """一键应用 Profile 优化建议（任务 17.7 / 需求 9.7 / 18.6）。

    管理员确认 ``GET /api/admin/feedback/suggestions`` 返回的优化建议后，
    通过本接口将建议中的字段（如 ``chunking`` / ``boilerplate`` /
    ``heading_rules``）落库：

    1. 调用 :meth:`FeedbackService.apply_profile_update` 写入更新并把
       ``profile.version`` +1（用于审计/前端版本展示）。
    2. 通过 :meth:`FeedbackService.get_affected_documents` 找出受影响文档，
       并提交 :meth:`FeedbackService.trigger_reprocessing` 异步批量重处理
       （返回 ``reprocessing_task_id`` 供后续进度查询）。

    错误处理：

    - 鉴权：``require_admin`` 守门，未登录返回 401，登录但非管理员 403。
    - Profile 不存在：返回 404 ``Profile not found``。
    """
    service = FeedbackService(db)

    profile = await service.apply_profile_update(
        profile_id=request.profile_id,
        updates=request.updates,
    )
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    # 触发受影响文档的批量重处理（任务 17.8 的入口，本任务仅串联触发逻辑）
    documents = await service.get_affected_documents(profile_id=request.profile_id)
    task_id = None
    if documents:
        doc_ids = [str(d.id) for d in documents]
        task = await service.trigger_reprocessing(doc_ids)
        task_id = task.task_id

    return ApplyResponse(
        success=True,
        message=f"Profile '{profile.name}' updated to version {profile.version}. "
        f"{len(documents)} documents queued for reprocessing.",
        reprocessing_task_id=task_id,
    )


@router.post("/api/admin/feedback/apply/dictionary", response_model=ApplyResponse)
async def apply_dictionary_suggestion(
    request: ApplyDictionaryRequest,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> ApplyResponse:
    """一键应用领域词典更新（任务 17.7 / 需求 9.7 / 18.6）。

    用于把建议中的 ``add_term`` 类型动作落库：

    1. 调用 :meth:`FeedbackService.apply_dictionary_update`，将 ``new_terms``
       合并到目标词典；服务层按 ``word`` 字段去重，避免重复术语。
    2. 触发关联文档的异步批量重处理，返回 ``reprocessing_task_id``。

    错误处理：

    - 鉴权：``require_admin`` 守门，未登录返回 401，登录但非管理员 403。
    - 词典不存在：返回 404 ``Dictionary not found``。
    """
    service = FeedbackService(db)

    dictionary = await service.apply_dictionary_update(
        dictionary_id=request.dictionary_id,
        new_terms=request.new_terms,
    )
    if not dictionary:
        raise HTTPException(status_code=404, detail="Dictionary not found")

    # 触发受影响文档的批量重处理
    documents = await service.get_affected_documents(dictionary_id=request.dictionary_id)
    task_id = None
    if documents:
        doc_ids = [str(d.id) for d in documents]
        task = await service.trigger_reprocessing(doc_ids)
        task_id = task.task_id

    return ApplyResponse(
        success=True,
        message=f"Dictionary '{dictionary.name}' updated with {len(request.new_terms)} terms. "
        f"{len(documents)} documents queued for reprocessing.",
        reprocessing_task_id=task_id,
    )


# ─── Reprocessing Progress ────────────────────────────────────────────


@router.post("/api/admin/feedback/reprocess", response_model=ReprocessProgressResponse)
async def trigger_reprocessing(
    request: ReprocessRequest,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> ReprocessProgressResponse:
    """触发受影响文档的批量重处理（任务 17.8 / 需求 9.8）。

    入参三选一（按优先级）：

    - ``document_ids``：显式给定的文档 ID 列表，直接入队，不再查询受影响范围。
    - ``profile_id``：先调用 :meth:`FeedbackService.get_affected_documents`
      解析 Profile 关联的所有已完成文档。
    - ``dictionary_id``：解析 Dictionary 关联 Profile 下的所有已完成文档。

    三者均未提供 → 422 ``Must provide profile_id, dictionary_id, or document_ids``。
    解析后没有任何受影响文档 → 404 ``No documents found for reprocessing``。

    成功路径：

    1. :meth:`FeedbackService.trigger_reprocessing` 提交 Celery 任务
       ``app.tasks.reprocess_document``（队列 ``reprocessing``），每篇文档独立
       入队，处理结果不阻塞其它文档（需求 9.8）。
    2. 同步把任务总量 / 状态 / 创建时间写入 Redis hash ``reprocess:task:{id}``，
       TTL 24h，供后续 ``GET /api/admin/feedback/reprocess/{task_id}`` 查询。

    鉴权：``require_admin`` 守门，未登录返回 401，登录但非管理员返回 403。
    """
    service = FeedbackService(db)

    if request.document_ids:
        doc_ids = request.document_ids
    elif request.profile_id or request.dictionary_id:
        documents = await service.get_affected_documents(
            profile_id=request.profile_id,
            dictionary_id=request.dictionary_id,
        )
        doc_ids = [str(d.id) for d in documents]
    else:
        raise HTTPException(
            status_code=422,
            detail="Must provide profile_id, dictionary_id, or document_ids",
        )

    if not doc_ids:
        raise HTTPException(
            status_code=404,
            detail="No documents found for reprocessing",
        )

    task = await service.trigger_reprocessing(doc_ids)

    progress_percent = 0.0
    if task.total_documents > 0:
        progress_percent = (task.processed_documents / task.total_documents) * 100

    return ReprocessProgressResponse(
        task_id=task.task_id,
        total_documents=task.total_documents,
        processed_documents=task.processed_documents,
        status=task.status,
        progress_percent=progress_percent,
        created_at=task.created_at,
        error=task.error,
    )


@router.get(
    "/api/admin/feedback/reprocess/{task_id}",
    response_model=ReprocessProgressResponse,
)
async def get_reprocessing_progress(
    task_id: str,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> ReprocessProgressResponse:
    """查询批量重处理任务进度（任务 17.8 / 需求 9.8）。

    从 Redis hash ``reprocess:task:{task_id}`` 读取最新进度，包括 ``status``、
    已处理文档数与总量，由前端按需轮询。

    - 任务不存在或已过期（24h TTL 失效）→ 404 ``Reprocessing task not found``。
    - 鉴权：``require_admin`` 守门，未登录返回 401，登录但非管理员返回 403。
    """
    service = FeedbackService(db)
    task = await service.get_reprocessing_progress(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Reprocessing task not found")

    progress_percent = 0.0
    if task.total_documents > 0:
        progress_percent = (task.processed_documents / task.total_documents) * 100

    return ReprocessProgressResponse(
        task_id=task.task_id,
        total_documents=task.total_documents,
        processed_documents=task.processed_documents,
        status=task.status,
        progress_percent=progress_percent,
        created_at=task.created_at,
        error=task.error,
    )
