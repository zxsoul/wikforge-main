"""Admin API for document review queue management.

Provides:
- List pending reviews (sorted by score, filterable by profile/space)
- Side-by-side preview (original file URL + parsed Markdown)
- Submit corrections (corrected Markdown, triggers re-chunking/vectorization)
- Approve/reject reviews
- Correction sample collection for profile optimization
"""

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import Float, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import require_admin
from app.core.database import get_db
from app.core.minio import generate_presigned_get_url
from app.models.document import Document
from app.models.document_profile import DocumentProfile
from app.models.document_review import DocumentReview, ReviewStatus
from app.models.user import User
from app.tasks.pipeline import submit_reprocess_from_markdown

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/reviews", tags=["admin-reviews"])


# ─── Request/Response Schemas ──────────────────────────────────────────


class QualityScoreResponse(BaseModel):
    """Quality score breakdown."""

    overall: float
    components: dict[str, float] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)


class ReviewListItem(BaseModel):
    """Single item in the review list (任务 11.9)。

    字段命名遵守任务说明：``review_id`` / ``document_title`` / ``profile_name``
    等便于前端审核队列页直接渲染。``quality_score`` 保留 JSONB 原貌，让前端
    可同时展示综合分与各维度子分。
    """

    review_id: str
    document_id: str
    document_title: str
    space_id: str
    profile_id: str | None = None
    profile_name: str | None = None
    quality_score: dict = Field(default_factory=dict)
    status: str
    created_at: datetime
    reviewed_at: datetime | None = None

    model_config = {"from_attributes": True}


class ReviewListResponse(BaseModel):
    """Paginated review list response。

    分页元数据使用 ``page`` / ``page_size`` / ``total``，与任务说明保持一致；
    保留 ``items`` 字段（任务说明指定）。
    """

    items: list[ReviewListItem]
    page: int
    page_size: int
    total: int


class DocumentPreviewResponse(BaseModel):
    """Side-by-side preview: original file URL + parsed Markdown."""

    review_id: str
    document_id: str
    document_title: str
    original_file_url: str
    parsed_markdown: str
    quality_score: QualityScoreResponse
    status: str


class CorrectionRequest(BaseModel):
    """Request to submit a corrected Markdown version.

    ``corrected_markdown`` 上限 5 MB——足以容纳一份较大 Wiki 文档的 Markdown
    源（典型企业知识库单篇 < 200 KB），同时挡掉明显的滥用 / 误传整文件二
    进制内容的场景。Pydantic 在请求阶段就拒绝过大 payload，避免到 DB JSONB
    写入阶段才报错。
    """

    corrected_markdown: str = Field(..., min_length=1, max_length=5 * 1024 * 1024)
    reviewer_note: str | None = Field(None, max_length=2000)


class CorrectionResponse(BaseModel):
    """Response after submitting a correction."""

    review_id: str
    status: str
    message: str


class ApproveRequest(BaseModel):
    """Request to approve a review."""

    reviewer_note: str | None = None


class CorrectionSample(BaseModel):
    """A correction sample for profile optimization (任务 11.12)。

    设计参考：design.md 「Feedback Loop」+ requirements.md 需求 17.7
    （收集修正数据作为 Profile 优化样本：原文本、修正后文本、使用的 Profile）。

    本模型对外暴露：

    - ``id`` / ``document_id`` / ``space_id``：定位样本与所属空间
    - ``profile_id`` / ``profile_name``：哪个 Profile 处理出的低质量解析。
      ``profile_name`` 通过 ``DocumentProfile`` 外连接补齐，便于后续聚合统计
      （「Profile X 修正样本数 / 关键问题」），文档没有匹配 Profile 时为
      ``None``（兜底场景，例如走了 Universal Parser）。
    - ``original_text`` / ``corrected_text``：解析阶段产出 vs 审核员修正后
      的 Markdown，可直接做 diff 喂给 Profile 优化器。
    - ``quality_score_snapshot``：修正发生时刻的质量分快照（``overall`` +
      ``components`` + ``issues``）。Profile 优化分析需要把「Profile + 评分
      构成 + 修正动作」三者关联，否则只看修正前后无法定位是哪个评分维度
      触发的低分（例如 ``heading_detection`` 低 → 标题层级在 Profile 里没
      配对）。
    - ``reviewed_at`` / ``corrected_at``：两个字段语义相同（``corrected_at``
      是审核员视角的别名），同时返回让前端两套语境都能直接绑定。
    - ``reviewer_note``：审核员留下的修正意图说明，用于人工标注训练样本。
    """

    id: str
    document_id: str
    space_id: str | None = None
    profile_id: str | None = None
    profile_name: str | None = None
    original_text: str
    corrected_text: str
    quality_score_snapshot: dict = Field(default_factory=dict)
    reviewer_note: str | None = None
    reviewed_at: datetime
    corrected_at: datetime
    # 历史字段：保留以兼容已经在用 ``created_at`` 的前端代码。等于
    # ``reviewed_at``（修正发生时间），不再额外曝光 review 行的 created_at。
    created_at: datetime


class CorrectionSampleListResponse(BaseModel):
    """List of correction samples."""

    samples: list[CorrectionSample]
    total: int
    skip: int
    limit: int


# ─── Endpoints ─────────────────────────────────────────────────────────


# 排序模式 → SQLAlchemy ORDER BY 子句的工厂。``score_asc`` 把最差的文档
# 排在最前面，便于审核员先处理质量最低的；``created_at_desc`` 用于查看最新
# 提交。表达式通过 ``cast`` 把 JSONB ``->> 'overall'`` 取出的字符串转成
# float 进行排序，PostgreSQL 上等价于 ``(quality_score->>'overall')::float``。
def _quality_score_overall_expr():
    """Return the SQLAlchemy expression for ``quality_score->>'overall'`` as float."""
    return cast(DocumentReview.quality_score["overall"].astext, Float)


def _build_sort_clause(sort_by: str):
    """Map ``sort_by`` query param to ORDER BY columns.

    Returns a tuple of clauses so callers can apply them with
    ``query.order_by(*clauses)``. We always tie-break on ``created_at``
    descending to keep results deterministic when scores are equal.
    """
    if sort_by == "quality_score_asc":
        return (
            _quality_score_overall_expr().asc().nullslast(),
            DocumentReview.created_at.desc(),
        )
    if sort_by == "created_at_desc":
        return (DocumentReview.created_at.desc(),)
    raise HTTPException(
        status_code=400,
        detail=(
            f"Invalid sort_by: {sort_by!r}. "
            "Allowed values: 'quality_score_asc', 'created_at_desc'."
        ),
    )


def _coerce_uuid(value: str, field_name: str) -> uuid.UUID:
    """Parse a UUID query param, returning 400 on malformed input."""
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid {field_name}: {value!r}"
        ) from exc


def _coerce_datetime(value: str, field_name: str) -> datetime:
    """Parse an ISO 8601 datetime query param, returning 400 on bad input.

    接受常见的 ISO 8601 写法：

    - ``2024-06-01``（仅日期，按 UTC 0 点处理）
    - ``2024-06-01T12:00:00``（naive，按 UTC 处理）
    - ``2024-06-01T12:00:00Z`` / ``2024-06-01T12:00:00+08:00``（带时区）

    Python 3.11+ 的 ``datetime.fromisoformat`` 已支持 ``Z`` 后缀（CPython
    GH-80010）。对没有 tzinfo 的解析结果，统一附上 UTC，确保数据库 timestamptz
    比较有明确语义。
    """
    try:
        parsed = datetime.fromisoformat(value)
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid {field_name}: {value!r}. "
                "Expected ISO 8601 (e.g. '2024-06-01' or "
                "'2024-06-01T00:00:00Z')."
            ),
        ) from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


@router.get("", response_model=ReviewListResponse)
async def list_reviews(
    status: str = Query(
        "pending",
        description=(
            "Filter by review status. Defaults to 'pending'. "
            "Allowed: pending, approved, corrected, rejected."
        ),
    ),
    profile_id: str | None = Query(
        None, description="Filter by document.matched_profile_id (UUID)."
    ),
    space_id: str | None = Query(
        None, description="Filter by document.space_id (UUID)."
    ),
    sort_by: str = Query(
        "quality_score_asc",
        description=(
            "Sort order. 'quality_score_asc' (default) surfaces the worst "
            "documents first; 'created_at_desc' shows newest first."
        ),
    ),
    page: int = Query(1, ge=1, description="1-indexed page number."),
    page_size: int = Query(20, ge=1, le=100, description="Items per page (1-100)."),
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> ReviewListResponse:
    """List reviews in the review queue (任务 11.9)。

    管理员页用于人工审核：默认按 ``status='pending'`` + ``quality_score_asc``
    排序，把分数最低的文档排在最前面（design.md 「审核队列页」要求）。
    支持按 ``profile_id`` / ``space_id`` 过滤，按 ``page`` / ``page_size``
    分页。

    通过 ``require_admin`` 强制管理员权限：未登录返回 401，登录但非管理员
    返回 403（详见 :func:`app.api.auth.require_admin`）。
    """
    # ─── 校验 status ──────────────────────────────────────────────
    try:
        status_enum = ReviewStatus(status)
    except ValueError as exc:
        allowed = ", ".join(sorted(s.value for s in ReviewStatus))
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status: {status!r}. Allowed: {allowed}.",
        ) from exc

    # ─── 校验 UUID 过滤参数 ───────────────────────────────────────
    profile_uuid = _coerce_uuid(profile_id, "profile_id") if profile_id else None
    space_uuid = _coerce_uuid(space_id, "space_id") if space_id else None

    # ─── 校验 sort_by（提前于 DB 访问，便于在 400 路径不打 DB） ──
    sort_clauses = _build_sort_clause(sort_by)

    # ─── 构造查询：DocumentReview JOIN Document JOIN DocumentProfile ─
    # outerjoin 到 DocumentProfile：matched_profile_id 可能为 NULL（文档
    # 可能没有匹配到 Profile，例如 Universal Parser 兜底场景）。
    base_filters = [DocumentReview.status == status_enum]
    if profile_uuid is not None:
        base_filters.append(Document.matched_profile_id == profile_uuid)
    if space_uuid is not None:
        base_filters.append(Document.space_id == space_uuid)

    # ─── total: 单独 COUNT(*) 查询 ────────────────────────────────
    count_stmt = (
        select(func.count(DocumentReview.id))
        .select_from(DocumentReview)
        .join(Document, DocumentReview.document_id == Document.id)
        .where(*base_filters)
    )
    count_result = await db.execute(count_stmt)
    total = int(count_result.scalar() or 0)

    # ─── items: 主查询 + 排序 + 分页 ──────────────────────────────
    offset = (page - 1) * page_size

    stmt = (
        select(
            DocumentReview,
            Document.title,
            Document.space_id,
            Document.matched_profile_id,
            DocumentProfile.name,
        )
        .join(Document, DocumentReview.document_id == Document.id)
        .outerjoin(
            DocumentProfile,
            Document.matched_profile_id == DocumentProfile.id,
        )
        .where(*base_filters)
        .order_by(*sort_clauses)
        .offset(offset)
        .limit(page_size)
    )

    result = await db.execute(stmt)
    rows = result.all()

    items: list[ReviewListItem] = []
    for review, title, doc_space_id, doc_profile_id, profile_name in rows:
        items.append(
            ReviewListItem(
                review_id=str(review.id),
                document_id=str(review.document_id),
                document_title=title or "",
                space_id=str(doc_space_id) if doc_space_id else "",
                profile_id=str(doc_profile_id) if doc_profile_id else None,
                profile_name=profile_name,
                quality_score=review.quality_score or {},
                status=review.status.value,
                created_at=review.created_at,
                reviewed_at=review.reviewed_at,
            )
        )

    return ReviewListResponse(
        items=items,
        page=page,
        page_size=page_size,
        total=total,
    )


@router.get("/{review_id}/preview", response_model=DocumentPreviewResponse)
async def preview_review(
    review_id: str,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> DocumentPreviewResponse:
    """文档并排预览 API（任务 11.10）。

    管理员审核详情页用左右两栏分别展示「原文件」与「解析后 Markdown」
    （design.md 「审核队列页 / 审核详情页」一节）。本接口返回：

    - ``original_file_url``：MinIO 预签名 GET URL（默认 600 秒 TTL，由
      :func:`app.core.minio.generate_presigned_get_url` 生成）。生成失败
      或 MinIO 未就绪时退化为后端代理路径
      ``/api/documents/{document_id}/download``，前端无需感知差异。
    - ``parsed_markdown``：清洗 + 结构识别后的完整 Markdown，由管线
      ``process_document`` 在打分时通过
      :func:`_persist_document_quality_score` 写入
      ``DocumentReview.quality_score['parsed_markdown']``。还没跑完管线的
      历史数据没有该字段，本接口此时返回一个明确的占位文本，前端可据此
      展示「暂无解析结果」提示而不是 4xx。
    - ``quality_score``：综合分 + 各维度子分 + 问题列表（来自
      ``DocumentReview.quality_score`` JSONB 列）。

    通过 ``require_admin`` 强制管理员权限：未登录返回 401，登录但非管理员
    返回 403，详见 :func:`app.api.auth.require_admin`。
    """
    # 1) review_id 校验：非法 UUID 直接 400，避免到 DB 才报错。
    try:
        review_uuid = uuid.UUID(review_id)
    except (ValueError, AttributeError) as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid review_id: {review_id!r}"
        ) from exc

    # 2) 取 review。``DocumentReview.document`` 配置了 ``lazy='selectin'``，
    #    SELECT 时一并把关联 Document 加载出来，不需要再单独查一次。
    result = await db.execute(
        select(DocumentReview).where(DocumentReview.id == review_uuid)
    )
    review = result.scalar_one_or_none()
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")

    doc = review.document
    if not doc:
        # Document 被外键 ON DELETE CASCADE 删掉之后理论上不应再出现 review，
        # 但 selectin 也可能因为并发删除拿到 None；当作 404 处理而不是 500。
        raise HTTPException(status_code=404, detail="Associated document not found")

    # 3) 构造 original_file_url。MinIO 预签名 URL 失败时退化为后端下载路径，
    #    保证前端 iframe / object 标签总能拿到一个可用 src。
    storage_path = getattr(doc, "storage_path", "") or ""
    presigned_url = (
        generate_presigned_get_url(storage_path) if storage_path else None
    )
    original_file_url = presigned_url or f"/api/documents/{doc.id}/download"

    # 4) parsed_markdown 取自 quality_score JSONB（管线 process_document 在
    #    11.10 起会持久化 ProcessedDocument.markdown 到此处）。历史数据没该
    #    字段时给一个明确占位，避免前端拿到空字符串后无法区分「真的为空」与
    #    「字段缺失」。
    quality = review.quality_score or {}
    parsed_markdown = quality.get("parsed_markdown") or ""
    if not parsed_markdown:
        parsed_markdown = f"[文档「{doc.title}」的解析结果暂未存储]"

    # 5) 透出 quality_score 各维度。``QualityScoreResponse`` 字段缺省值能容忍
    #    JSONB 列里的部分缺失（例如旧版本仅写了 overall）。
    return DocumentPreviewResponse(
        review_id=str(review.id),
        document_id=str(doc.id),
        document_title=doc.title,
        original_file_url=original_file_url,
        parsed_markdown=parsed_markdown,
        quality_score=QualityScoreResponse(
            overall=float(quality.get("overall", 0.0) or 0.0),
            components=quality.get("components") or {},
            issues=quality.get("issues") or [],
        ),
        status=review.status.value,
    )


@router.post("/{review_id}/correct", response_model=CorrectionResponse)
async def submit_correction(
    review_id: str,
    request: CorrectionRequest,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> CorrectionResponse:
    """Submit a corrected Markdown version for a document（任务 11.11）。

    管理员审核详情页提交修正后调用：

    1. 校验 ``review_id`` 是合法 UUID（400），并且 ``corrected_markdown``
       去掉首尾空白后非空（400）；上限由 ``CorrectionRequest`` 字段约束
       拦在请求阶段。
    2. 仅 ``pending`` / ``rejected`` 状态可被修正（400 阻止重复修正或对
       已通过的审核改写）。
    3. 把修正前 Markdown 落到 ``original_markdown``、修正后 Markdown 落到
       ``corrected_markdown``、修正时间落到 ``correction_timestamp``，
       全部存进 ``DocumentReview.quality_score`` JSONB 列；状态置为
       ``corrected``。
    4. 调用 ``submit_reprocess_from_markdown`` 触发
       ``cleanup → chunk → embed → index`` 的 Celery chain，跳过
       parse / profile_match / process（修正后的内容已是 reviewer 认可的
       cleaned Markdown，再跑解析既浪费又会覆盖修正）。

    Celery 不可用时不会让 API 失败：``submit_reprocess_from_markdown`` 内部
    捕获异常并记 WARNING，本函数据返回值调整提示文案，让管理员仍能完成
    审核流程。

    通过 ``require_admin`` 强制管理员权限：未登录返回 401，登录但非管理员
    返回 403。
    """
    # 1) review_id 校验：非法 UUID 直接 400，避免到 DB 才报错。
    try:
        review_uuid = uuid.UUID(review_id)
    except (ValueError, AttributeError) as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid review_id: {review_id!r}"
        ) from exc

    # 2) 入参语义校验：``min_length=1`` 已挡空串，但用户可能提交全空白
    #    （``"   \n"``）。strip 后为空一律 400，避免把无意义内容写进 JSONB。
    corrected_markdown = request.corrected_markdown
    if not corrected_markdown.strip():
        raise HTTPException(
            status_code=400,
            detail="corrected_markdown cannot be empty or whitespace-only",
        )

    # 3) 取 review。``DocumentReview.document`` 上挂 ``lazy='selectin'``，
    #    SELECT 时一并加载关联 Document，无需第二次查询。
    result = await db.execute(
        select(DocumentReview).where(DocumentReview.id == review_uuid)
    )
    review = result.scalar_one_or_none()
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")

    if review.status not in (ReviewStatus.pending, ReviewStatus.rejected):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot correct a review with status '{review.status.value}'. "
            f"Only 'pending' or 'rejected' reviews can be corrected.",
        )

    # 4) 把修正前 Markdown 留底（管理员之后还能比对 / 后续优化 Profile）。
    #    ``parsed_markdown`` 由任务 11.10 在 process_document 阶段写入；
    #    历史数据可能为空字符串。
    quality = dict(review.quality_score or {})
    original_markdown = quality.get("parsed_markdown", "") or quality.get(
        "original_markdown", ""
    )

    # 5) 更新审核状态。``reviewer_note`` 透传到 DocumentReview.reviewer_note
    #    便于在审核列表里展示。
    review.status = ReviewStatus.corrected
    review.reviewer_note = request.reviewer_note
    review.reviewed_at = datetime.now(timezone.utc)

    quality["corrected_markdown"] = corrected_markdown
    quality["original_markdown"] = original_markdown
    quality["correction_timestamp"] = datetime.now(timezone.utc).isoformat()
    review.quality_score = quality

    await db.flush()
    await db.refresh(review)

    # 6) 触发 Celery 链。``submit_reprocess_from_markdown`` 内部已捕获
    #    Celery 不可用 / broker 离线场景，返回 False 表示「未触发」。我们
    #    据此调整提示文案，但 API 始终 200——审核操作不应因消息队列宕机而
    #    阻塞。
    try:
        triggered = submit_reprocess_from_markdown(
            str(review.document_id), corrected_markdown
        )
    except Exception as exc:  # noqa: BLE001 — defensive; pipeline import shouldn't fail
        logger.warning(
            "submit_reprocess_from_markdown raised for document %s: %s",
            review.document_id,
            exc,
        )
        triggered = False

    if triggered:
        message = "修正已提交，已重新触发分块和向量化流程"
    else:
        message = (
            "修正已提交，但消息队列不可用，请稍后由运维同步触发 reprocess"
        )

    logger.info(
        "Correction submitted for review %s, document %s. Reprocess triggered: %s",
        review_id,
        review.document_id,
        triggered,
    )

    return CorrectionResponse(
        review_id=str(review.id),
        status=review.status.value,
        message=message,
    )


@router.post("/{review_id}/approve", response_model=CorrectionResponse)
async def approve_review(
    review_id: str,
    request: ApproveRequest,
    db: AsyncSession = Depends(get_db),
) -> CorrectionResponse:
    """Approve a review, marking the document as acceptable quality."""
    result = await db.execute(
        select(DocumentReview).where(DocumentReview.id == uuid.UUID(review_id))
    )
    review = result.scalar_one_or_none()
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")

    if review.status != ReviewStatus.pending:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot approve a review with status '{review.status.value}'. "
            f"Only 'pending' reviews can be approved.",
        )

    review.status = ReviewStatus.approved
    review.reviewer_note = request.reviewer_note
    review.reviewed_at = datetime.now(timezone.utc)

    await db.flush()
    await db.refresh(review)

    return CorrectionResponse(
        review_id=str(review.id),
        status=review.status.value,
        message="审核已通过",
    )


@router.post("/{review_id}/reject", response_model=CorrectionResponse)
async def reject_review(
    review_id: str,
    request: ApproveRequest,
    db: AsyncSession = Depends(get_db),
) -> CorrectionResponse:
    """Reject a review, marking the document as unacceptable."""
    result = await db.execute(
        select(DocumentReview).where(DocumentReview.id == uuid.UUID(review_id))
    )
    review = result.scalar_one_or_none()
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")

    if review.status != ReviewStatus.pending:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot reject a review with status '{review.status.value}'. "
            f"Only 'pending' reviews can be rejected.",
        )

    review.status = ReviewStatus.rejected
    review.reviewer_note = request.reviewer_note
    review.reviewed_at = datetime.now(timezone.utc)

    await db.flush()
    await db.refresh(review)

    return CorrectionResponse(
        review_id=str(review.id),
        status=review.status.value,
        message="审核已驳回",
    )


@router.get("/samples", response_model=CorrectionSampleListResponse)
async def list_correction_samples(
    profile_id: str | None = Query(
        None, description="Filter by Document.matched_profile_id (UUID)."
    ),
    space_id: str | None = Query(
        None, description="Filter by Document.space_id (UUID)."
    ),
    date_from: str | None = Query(
        None,
        description=(
            "Inclusive lower bound on reviewed_at. ISO 8601, e.g. "
            "'2024-06-01T00:00:00Z' or '2024-06-01'. Naive values are "
            "interpreted as UTC."
        ),
    ),
    date_to: str | None = Query(
        None,
        description=(
            "Inclusive upper bound on reviewed_at. Same format as date_from."
        ),
    ),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> CorrectionSampleListResponse:
    """List correction samples for profile optimization (任务 11.12)。

    管理员后台用于调取「修正样本集」做 Profile 调优分析（design.md 「Feedback
    Loop」+ requirements.md 需求 17.7）。返回所有 ``status='corrected'`` 的
    审核记录，附带：

    - ``original_text`` / ``corrected_text``：原始解析与修正后 Markdown
      （存储在 ``DocumentReview.quality_score`` JSONB 列）。
    - ``profile_id`` / ``profile_name``：通过 ``Document.matched_profile_id``
      外连接 ``DocumentProfile`` 取得，便于按 Profile 维度聚合。
    - ``quality_score_snapshot``：修正发生时的 ``overall`` / ``components``
      / ``issues``，让分析端能把评分构成与修正动作关联起来。
    - ``reviewed_at`` / ``corrected_at`` / ``created_at``：均指向修正时间，
      多个字段为兼容多种前端语境。

    支持过滤 / 分页：

    - ``profile_id``：按 Profile 过滤
    - ``space_id``：按空间过滤
    - ``date_from`` / ``date_to``：按修正时间窗口（reviewed_at）过滤；
      ISO 8601 格式，naive 值按 UTC 处理；非法格式返回 400。
    - ``skip`` / ``limit``：偏移分页

    通过 ``require_admin`` 强制管理员权限：未登录返回 401，登录但非管理员
    返回 403。
    """
    # ─── 校验 UUID 过滤参数 ───────────────────────────────────────
    profile_uuid = _coerce_uuid(profile_id, "profile_id") if profile_id else None
    space_uuid = _coerce_uuid(space_id, "space_id") if space_id else None

    # ─── 校验 date 过滤参数 ───────────────────────────────────────
    date_from_dt = _coerce_datetime(date_from, "date_from") if date_from else None
    date_to_dt = _coerce_datetime(date_to, "date_to") if date_to else None

    # ─── 构造过滤条件 ─────────────────────────────────────────────
    base_filters = [DocumentReview.status == ReviewStatus.corrected]
    if profile_uuid is not None:
        base_filters.append(Document.matched_profile_id == profile_uuid)
    if space_uuid is not None:
        base_filters.append(Document.space_id == space_uuid)
    if date_from_dt is not None:
        base_filters.append(DocumentReview.reviewed_at >= date_from_dt)
    if date_to_dt is not None:
        base_filters.append(DocumentReview.reviewed_at <= date_to_dt)

    # ─── total: 单独 COUNT(*) 查询 ────────────────────────────────
    count_query = (
        select(func.count(DocumentReview.id))
        .select_from(DocumentReview)
        .join(Document, DocumentReview.document_id == Document.id)
        .where(*base_filters)
    )
    count_result = await db.execute(count_query)
    total = int(count_result.scalar() or 0)

    # ─── 主查询：JOIN Document + outerjoin DocumentProfile ────────
    # ``matched_profile_id`` 可能为空（兜底走 Universal Parser 的场景），
    # 所以 outerjoin 保留 NULL 行；列表里 ``profile_name`` 字段就是 None。
    stmt = (
        select(
            DocumentReview,
            Document.space_id,
            Document.matched_profile_id,
            DocumentProfile.name,
        )
        .join(Document, DocumentReview.document_id == Document.id)
        .outerjoin(
            DocumentProfile,
            Document.matched_profile_id == DocumentProfile.id,
        )
        .where(*base_filters)
        .order_by(DocumentReview.reviewed_at.desc())
        .offset(skip)
        .limit(limit)
    )
    result = await db.execute(stmt)
    rows = result.all()

    samples: list[CorrectionSample] = []
    for review, doc_space_id, doc_profile_id, profile_name in rows:
        quality = review.quality_score or {}
        # ``reviewed_at`` 是修正动作发生的时间；旧数据可能为空（理论上不
        # 应出现，因为只筛选 status='corrected'，但万一脏数据，退化到
        # created_at，避免 Pydantic 报错。
        reviewed_at = review.reviewed_at or review.created_at

        # 质量分快照：只保留 overall / components / issues，剔除原始/修正
        # Markdown（避免 quality_score_snapshot 字段重复放大返回 payload）。
        snapshot = {
            "overall": float(quality.get("overall", 0.0) or 0.0),
            "components": quality.get("components") or {},
            "issues": quality.get("issues") or [],
        }

        samples.append(
            CorrectionSample(
                id=str(review.id),
                document_id=str(review.document_id),
                space_id=str(doc_space_id) if doc_space_id else None,
                profile_id=str(doc_profile_id) if doc_profile_id else None,
                profile_name=profile_name,
                original_text=quality.get("original_markdown", "") or "",
                corrected_text=quality.get("corrected_markdown", "") or "",
                quality_score_snapshot=snapshot,
                reviewer_note=review.reviewer_note,
                reviewed_at=reviewed_at,
                corrected_at=reviewed_at,
                created_at=reviewed_at,
            )
        )

    return CorrectionSampleListResponse(
        samples=samples, total=total, skip=skip, limit=limit
    )
