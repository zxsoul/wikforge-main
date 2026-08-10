"""Feedback service for search quality feedback collection, aggregation, and optimization.

Provides:
- Feedback CRUD (create, list, get)
- Aggregation analysis (by profile, document, query type, time range)
- Error pattern detection (repeated errors under same profile)
- Optimization suggestion generation (adjust_chunking, add_term, update_boilerplate, etc.)
- Apply suggestions (update Profile/Dictionary, trigger reprocessing)
- Batch reprocessing progress tracking
"""

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.document_profile import DocumentProfile
from app.models.domain_dictionary import DomainDictionary
from app.models.search_feedback import SearchFeedback

logger = logging.getLogger(__name__)


# ─── Enums & Constants ─────────────────────────────────────────────────


class FeedbackType(str, Enum):
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    ISSUE = "issue"


class IssueCategory(str, Enum):
    """检索反馈的「问题类型」枚举（需求 9.2 / 18.2）。

    与 design.md 定义保持一致，用于：
    - ``SearchFeedback.issue_category`` 字段写入校验
    - ``GET /api/feedback/issue-categories`` 给前端下拉框使用
    """

    IRRELEVANT = "irrelevant"
    MISSING_INFO = "missing_info"
    CITATION_ERROR = "citation_error"
    FORMAT = "format"
    OTHER = "other"


# 问题类型的中文标签映射，供前端下拉框直接渲染。
# 显式集中维护，避免 i18n 文案散落在多个文件中。
ISSUE_CATEGORY_LABELS: dict[str, str] = {
    IssueCategory.IRRELEVANT.value: "结果不相关",
    IssueCategory.MISSING_INFO.value: "缺少关键信息",
    IssueCategory.CITATION_ERROR.value: "引用错误",
    IssueCategory.FORMAT.value: "格式问题",
    IssueCategory.OTHER.value: "其他",
}


class SuggestionType(str, Enum):
    ADJUST_CHUNKING = "adjust_chunking"
    ADD_TERM = "add_term"
    UPDATE_BOILERPLATE = "update_boilerplate"
    ADJUST_HEADING_RULES = "adjust_heading_rules"
    ADD_QUERY_TEMPLATE = "add_query_template"


# Minimum number of same-type errors under a profile to trigger a suggestion
PATTERN_DETECTION_THRESHOLD = 3


# ─── Data Structures ───────────────────────────────────────────────────


@dataclass
class FeedbackFilter:
    """Filter criteria for feedback queries."""

    profile_id: str | None = None
    document_id: str | None = None
    feedback_type: str | None = None
    issue_category: str | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    user_id: str | None = None


@dataclass
class FeedbackAggregation:
    """Aggregated feedback statistics.

    支持按以下维度聚合反馈（需求 9.4 / 18.4）：

    - ``by_issue_category``：按问题类型聚合（仅 ``feedback_type == issue`` 的样本）。
    - ``by_profile``：按 ``related_profile_id`` 聚合。
    - ``by_document``：按 ``returned_results`` 中的结果标识（chunk / document
      ID 由调用方约定）聚合，统计每个结果被反馈触达的次数；同一条反馈中重复
      出现的同一标识只计 1 次（避免 returned_results 内部重复放大计数）。
    - ``by_date``：按反馈创建日期（``YYYY-MM-DD``）聚合，用于时间范围趋势。
    """

    total_count: int = 0
    thumbs_up_count: int = 0
    thumbs_down_count: int = 0
    issue_count: int = 0
    by_issue_category: dict[str, int] = field(default_factory=dict)
    by_profile: dict[str, int] = field(default_factory=dict)
    by_document: dict[str, int] = field(default_factory=dict)
    by_date: dict[str, int] = field(default_factory=dict)


@dataclass
class ErrorPattern:
    """Detected error pattern in feedback data."""

    profile_id: str
    profile_name: str
    issue_category: str
    occurrence_count: int
    sample_queries: list[str] = field(default_factory=list)
    first_seen: datetime | None = None
    last_seen: datetime | None = None


@dataclass
class OptimizationSuggestion:
    """Generated optimization suggestion based on error patterns."""

    type: str
    target_id: str
    target_name: str
    recommendation: dict = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    description: str = ""


@dataclass
class ReprocessingTask:
    """Tracks a batch reprocessing task."""

    task_id: str
    profile_id: str | None = None
    dictionary_id: str | None = None
    total_documents: int = 0
    processed_documents: int = 0
    status: str = "pending"  # pending, running, completed, failed
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error: str | None = None


# ─── Feedback Service ──────────────────────────────────────────────────


class FeedbackService:
    """Service for feedback collection, analysis, and optimization."""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ─── CRUD ──────────────────────────────────────────────────────

    async def create_feedback(
        self,
        user_id: uuid.UUID,
        query: str,
        returned_results: list[str],
        feedback_type: str,
        issue_category: str | None = None,
        comment: str | None = None,
        related_profile_id: str | None = None,
    ) -> SearchFeedback:
        """Create a new feedback entry.

        Args:
            user_id: The user providing feedback
            query: The search query that was executed
            returned_results: List of chunk IDs returned
            feedback_type: One of thumbs_up, thumbs_down, issue
            issue_category: Required when feedback_type is 'issue'
            comment: Optional text description (max 500 chars)
            related_profile_id: Optional associated profile ID

        Returns:
            The created SearchFeedback record

        Raises:
            ValueError: If feedback_type or issue_category is invalid
        """
        # Validate feedback_type
        valid_types = [ft.value for ft in FeedbackType]
        if feedback_type not in valid_types:
            raise ValueError(
                f"Invalid feedback_type '{feedback_type}'. Must be one of: {valid_types}"
            )

        # Validate issue_category when feedback_type is 'issue'
        if feedback_type == FeedbackType.ISSUE:
            valid_categories = [ic.value for ic in IssueCategory]
            if not issue_category:
                raise ValueError(
                    "issue_category is required when feedback_type is 'issue'"
                )
            if issue_category not in valid_categories:
                raise ValueError(
                    f"Invalid issue_category '{issue_category}'. "
                    f"Must be one of: {valid_categories}"
                )

        # Validate comment length
        if comment and len(comment) > 500:
            raise ValueError("Comment must not exceed 500 characters")

        feedback = SearchFeedback(
            user_id=user_id,
            query=query,
            returned_results=returned_results,
            feedback_type=feedback_type,
            issue_category=issue_category,
            comment=comment,
            related_profile_id=(
                uuid.UUID(related_profile_id) if related_profile_id else None
            ),
        )
        self.db.add(feedback)
        await self.db.flush()
        await self.db.refresh(feedback)
        return feedback

    async def get_feedback(self, feedback_id: str) -> SearchFeedback | None:
        """Get a single feedback entry by ID."""
        result = await self.db.execute(
            select(SearchFeedback).where(
                SearchFeedback.id == uuid.UUID(feedback_id)
            )
        )
        return result.scalar_one_or_none()

    async def list_feedbacks(
        self,
        filter: FeedbackFilter | None = None,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[list[SearchFeedback], int]:
        """List feedbacks with optional filtering."""
        query = select(SearchFeedback)
        query = self._apply_filters(query, filter)

        # Count total
        count_query = select(func.count(SearchFeedback.id))
        count_query = self._apply_filters(count_query, filter)
        count_result = await self.db.execute(count_query)
        total = count_result.scalar() or 0

        # Paginate
        query = query.order_by(SearchFeedback.created_at.desc())
        query = query.offset(skip).limit(limit)
        result = await self.db.execute(query)
        feedbacks = list(result.scalars().all())

        return feedbacks, total

    # ─── Aggregation ───────────────────────────────────────────────

    async def aggregate_feedback(
        self, filter: FeedbackFilter | None = None
    ) -> FeedbackAggregation:
        """Aggregate feedback data by various dimensions.

        支持按以下维度聚合（需求 9.4 / 18.4）：

        - 反馈类型（``thumbs_up`` / ``thumbs_down`` / ``issue``）总量
        - 问题类型（``issue_category``）
        - Profile（``related_profile_id``）
        - 文档 / Chunk 维度（``returned_results``）
        - 时间范围（按日聚合，配合 ``filter.start_date`` / ``filter.end_date``
          实现时间窗过滤）

        ``filter.document_id`` 用于把聚合范围限定到单篇文档：仅统计
        ``returned_results`` 中包含该 ID 的反馈样本。
        """
        query = select(SearchFeedback)
        query = self._apply_filters(query, filter)
        result = await self.db.execute(query)
        feedbacks = list(result.scalars().all())

        aggregation = FeedbackAggregation()
        aggregation.total_count = len(feedbacks)

        for fb in feedbacks:
            # Count by type
            if fb.feedback_type == FeedbackType.THUMBS_UP:
                aggregation.thumbs_up_count += 1
            elif fb.feedback_type == FeedbackType.THUMBS_DOWN:
                aggregation.thumbs_down_count += 1
            elif fb.feedback_type == FeedbackType.ISSUE:
                aggregation.issue_count += 1

            # Count by issue category
            if fb.issue_category:
                aggregation.by_issue_category[fb.issue_category] = (
                    aggregation.by_issue_category.get(fb.issue_category, 0) + 1
                )

            # Count by profile
            if fb.related_profile_id:
                profile_key = str(fb.related_profile_id)
                aggregation.by_profile[profile_key] = (
                    aggregation.by_profile.get(profile_key, 0) + 1
                )

            # Count by document / chunk reference inside returned_results。
            # 同一反馈内重复出现的同一 ID 仅计 1 次，防止前端把 chunk 列表
            # 里偶发的重复条目放大成「该文档被多次反馈」。
            if fb.returned_results:
                seen_in_feedback: set[str] = set()
                for ref in fb.returned_results:
                    if not isinstance(ref, str) or not ref:
                        continue
                    if ref in seen_in_feedback:
                        continue
                    seen_in_feedback.add(ref)
                    aggregation.by_document[ref] = (
                        aggregation.by_document.get(ref, 0) + 1
                    )

            # Count by date
            date_key = fb.created_at.strftime("%Y-%m-%d")
            aggregation.by_date[date_key] = (
                aggregation.by_date.get(date_key, 0) + 1
            )

        return aggregation

    # ─── Pattern Detection ─────────────────────────────────────────

    async def detect_error_patterns(
        self,
        min_occurrences: int = PATTERN_DETECTION_THRESHOLD,
        days_lookback: int = 30,
    ) -> list[ErrorPattern]:
        """Detect repeated error patterns in feedback data.

        Identifies cases where the same issue_category appears multiple times
        under the same profile, indicating a systematic problem.

        Args:
            min_occurrences: Minimum number of same-type errors to flag
            days_lookback: Number of days to look back for patterns

        Returns:
            List of detected error patterns
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_lookback)

        # Query negative feedback with profile association
        query = (
            select(SearchFeedback)
            .where(
                and_(
                    SearchFeedback.feedback_type.in_([
                        FeedbackType.THUMBS_DOWN,
                        FeedbackType.ISSUE,
                    ]),
                    SearchFeedback.related_profile_id.isnot(None),
                    SearchFeedback.created_at >= cutoff_date,
                )
            )
            .order_by(SearchFeedback.created_at.desc())
        )
        result = await self.db.execute(query)
        feedbacks = list(result.scalars().all())

        # Group by (profile_id, issue_category)
        pattern_groups: dict[tuple[str, str], list[SearchFeedback]] = defaultdict(list)
        for fb in feedbacks:
            category = fb.issue_category or "general_negative"
            key = (str(fb.related_profile_id), category)
            pattern_groups[key].append(fb)

        # Detect patterns exceeding threshold
        patterns: list[ErrorPattern] = []
        for (profile_id, category), group_feedbacks in pattern_groups.items():
            if len(group_feedbacks) >= min_occurrences:
                # Get profile name
                profile_name = await self._get_profile_name(profile_id)

                # Collect sample queries (up to 5)
                sample_queries = [
                    fb.query for fb in group_feedbacks[:5]
                ]

                timestamps = [fb.created_at for fb in group_feedbacks]
                pattern = ErrorPattern(
                    profile_id=profile_id,
                    profile_name=profile_name or "Unknown",
                    issue_category=category,
                    occurrence_count=len(group_feedbacks),
                    sample_queries=sample_queries,
                    first_seen=min(timestamps) if timestamps else None,
                    last_seen=max(timestamps) if timestamps else None,
                )
                patterns.append(pattern)

        # Sort by occurrence count descending
        patterns.sort(key=lambda p: p.occurrence_count, reverse=True)
        return patterns

    # ─── Suggestion Generation ─────────────────────────────────────

    async def generate_suggestions(
        self,
        patterns: list[ErrorPattern] | None = None,
    ) -> list[OptimizationSuggestion]:
        """Generate optimization suggestions based on detected error patterns.

        Maps error patterns to actionable suggestions:
        - irrelevant → adjust_chunking (chunk size may be too large/small)
        - missing_info → adjust_chunking (overlap may be insufficient)
        - citation_error → update_boilerplate (noise not properly removed)
        - format → adjust_heading_rules (structure not properly detected)
        - general_negative with high frequency → add_term (terminology issues)

        Args:
            patterns: Pre-detected patterns, or None to detect fresh

        Returns:
            List of optimization suggestions
        """
        if patterns is None:
            patterns = await self.detect_error_patterns()

        suggestions: list[OptimizationSuggestion] = []

        for pattern in patterns:
            suggestion = self._pattern_to_suggestion(pattern)
            if suggestion:
                suggestions.append(suggestion)

        return suggestions

    # ─── Apply Suggestions ─────────────────────────────────────────

    async def apply_profile_update(
        self,
        profile_id: str,
        updates: dict,
    ) -> DocumentProfile | None:
        """Apply optimization updates to a DocumentProfile.

        Args:
            profile_id: The profile to update
            updates: Dict of fields to update (chunking, boilerplate, heading_rules, etc.)

        Returns:
            Updated profile or None if not found
        """
        result = await self.db.execute(
            select(DocumentProfile).where(
                DocumentProfile.id == uuid.UUID(profile_id)
            )
        )
        profile = result.scalar_one_or_none()
        if not profile:
            return None

        # Apply updates
        if "chunking" in updates:
            profile.chunking = updates["chunking"]
        if "boilerplate" in updates:
            profile.boilerplate = updates["boilerplate"]
        if "heading_rules" in updates:
            profile.heading_rules = updates["heading_rules"]

        profile.version += 1
        await self.db.flush()
        await self.db.refresh(profile)

        logger.info(f"Profile '{profile.name}' updated to version {profile.version}")
        return profile

    async def apply_dictionary_update(
        self,
        dictionary_id: str,
        new_terms: list[dict],
    ) -> DomainDictionary | None:
        """Add terms to a DomainDictionary.

        Args:
            dictionary_id: The dictionary to update
            new_terms: List of term dicts to add ({"word": ..., "pos": ..., "weight": ...})

        Returns:
            Updated dictionary or None if not found
        """
        result = await self.db.execute(
            select(DomainDictionary).where(
                DomainDictionary.id == uuid.UUID(dictionary_id)
            )
        )
        dictionary = result.scalar_one_or_none()
        if not dictionary:
            return None

        # Merge terms (avoid duplicates)
        existing_words = {
            t["word"] if isinstance(t, dict) else t
            for t in (dictionary.terms or [])
        }
        current_terms = list(dictionary.terms or [])

        for term_data in new_terms:
            word = term_data.get("word", "")
            if word and word not in existing_words:
                current_terms.append(term_data)
                existing_words.add(word)

        dictionary.terms = current_terms
        await self.db.flush()
        await self.db.refresh(dictionary)

        logger.info(
            f"Dictionary '{dictionary.name}' updated with {len(new_terms)} new terms"
        )
        return dictionary

    # ─── Reprocessing ──────────────────────────────────────────────

    async def get_affected_documents(
        self,
        profile_id: str | None = None,
        dictionary_id: str | None = None,
    ) -> list[Document]:
        """Get documents affected by a profile or dictionary update.

        Args:
            profile_id: If provided, find documents using this profile
            dictionary_id: If provided, find documents using profiles linked to this dictionary

        Returns:
            List of affected documents
        """
        if profile_id:
            # Find documents that were processed with this profile
            # Documents are linked to profiles via their processing metadata
            query = select(Document).where(
                Document.status == "completed"
            )
            result = await self.db.execute(query)
            documents = list(result.scalars().all())
            return documents

        if dictionary_id:
            # Find profiles using this dictionary, then their documents
            profile_query = select(DocumentProfile).where(
                DocumentProfile.domain_dictionary_id == uuid.UUID(dictionary_id)
            )
            profile_result = await self.db.execute(profile_query)
            profiles = list(profile_result.scalars().all())

            if not profiles:
                return []

            # Get all completed documents (in a real system, we'd track which
            # profile was used for each document)
            query = select(Document).where(Document.status == "completed")
            result = await self.db.execute(query)
            return list(result.scalars().all())

        return []

    async def trigger_reprocessing(
        self,
        document_ids: list[str],
    ) -> ReprocessingTask:
        """Trigger async reprocessing of affected documents.

        Creates a batch reprocessing task and submits documents to the Celery queue.

        Args:
            document_ids: List of document IDs to reprocess

        Returns:
            ReprocessingTask with tracking information
        """
        task_id = str(uuid.uuid4())
        task = ReprocessingTask(
            task_id=task_id,
            total_documents=len(document_ids),
            processed_documents=0,
            status="pending",
        )

        # Submit reprocessing tasks to Celery
        try:
            from app.core.celery_app import celery_app

            for doc_id in document_ids:
                celery_app.send_task(
                    "app.tasks.reprocess_document",
                    args=[doc_id, task_id],
                    queue="reprocessing",
                )

            task.status = "running"
            logger.info(
                f"Reprocessing task {task_id} started for {len(document_ids)} documents"
            )
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            logger.error(f"Failed to trigger reprocessing: {e}")

        # Store task status in Redis for progress tracking
        try:
            from app.core.redis import get_redis

            redis = await get_redis()
            await redis.hset(
                f"reprocess:task:{task_id}",
                mapping={
                    "total": str(task.total_documents),
                    "processed": "0",
                    "status": task.status,
                    "created_at": task.created_at.isoformat(),
                    "error": task.error or "",
                },
            )
            await redis.expire(f"reprocess:task:{task_id}", 86400)  # 24h TTL
        except Exception as e:
            logger.warning(f"Failed to store reprocessing task in Redis: {e}")

        return task

    async def get_reprocessing_progress(
        self, task_id: str
    ) -> ReprocessingTask | None:
        """Get the progress of a reprocessing task.

        Args:
            task_id: The task ID to query

        Returns:
            ReprocessingTask with current progress, or None if not found
        """
        try:
            from app.core.redis import get_redis

            redis = await get_redis()
            data = await redis.hgetall(f"reprocess:task:{task_id}")

            if not data:
                return None

            return ReprocessingTask(
                task_id=task_id,
                total_documents=int(data.get("total", 0)),
                processed_documents=int(data.get("processed", 0)),
                status=data.get("status", "unknown"),
                created_at=datetime.fromisoformat(data["created_at"])
                if "created_at" in data
                else datetime.now(timezone.utc),
                error=data.get("error") or None,
            )
        except Exception as e:
            logger.warning(f"Failed to get reprocessing progress: {e}")
            return None

    # ─── Private Helpers ───────────────────────────────────────────

    def _apply_filters(self, query, filter: FeedbackFilter | None):
        """Apply filter criteria to a SQLAlchemy query."""
        if not filter:
            return query

        conditions = []
        if filter.profile_id:
            conditions.append(
                SearchFeedback.related_profile_id == uuid.UUID(filter.profile_id)
            )
        if filter.feedback_type:
            conditions.append(SearchFeedback.feedback_type == filter.feedback_type)
        if filter.issue_category:
            conditions.append(SearchFeedback.issue_category == filter.issue_category)
        if filter.start_date:
            conditions.append(SearchFeedback.created_at >= filter.start_date)
        if filter.end_date:
            conditions.append(SearchFeedback.created_at <= filter.end_date)
        if filter.user_id:
            conditions.append(
                SearchFeedback.user_id == uuid.UUID(filter.user_id)
            )
        if filter.document_id:
            # ``returned_results`` 是 JSONB 数组（chunk / document ID 列表）。
            # PostgreSQL 通过 ``@>`` 判断「数组包含目标元素」，对应 SQLAlchemy
            # 的 ``contains([value])``；JSON 字面量需要保留为列表而非裸字符串。
            conditions.append(
                SearchFeedback.returned_results.contains([filter.document_id])
            )

        if conditions:
            query = query.where(and_(*conditions))

        return query

    async def _get_profile_name(self, profile_id: str) -> str | None:
        """Get profile name by ID."""
        result = await self.db.execute(
            select(DocumentProfile.name).where(
                DocumentProfile.id == uuid.UUID(profile_id)
            )
        )
        row = result.scalar_one_or_none()
        return row if row else None

    def _pattern_to_suggestion(
        self, pattern: ErrorPattern
    ) -> OptimizationSuggestion | None:
        """Convert an error pattern to an optimization suggestion."""
        category = pattern.issue_category
        confidence = min(pattern.occurrence_count / 10.0, 1.0)

        if category == IssueCategory.IRRELEVANT:
            return OptimizationSuggestion(
                type=SuggestionType.ADJUST_CHUNKING,
                target_id=pattern.profile_id,
                target_name=pattern.profile_name,
                recommendation={
                    "action": "reduce_max_tokens",
                    "current_issue": "Chunks may be too large, including irrelevant content",
                    "suggested_max_tokens": 512,
                    "suggested_overlap_tokens": 100,
                },
                evidence=[f"query: {q}" for q in pattern.sample_queries],
                confidence=confidence,
                description=(
                    f"Profile '{pattern.profile_name}' 下有 {pattern.occurrence_count} 次"
                    f"'结果不相关'反馈，建议减小分块大小以提高精确度"
                ),
            )

        elif category == IssueCategory.MISSING_INFO:
            return OptimizationSuggestion(
                type=SuggestionType.ADJUST_CHUNKING,
                target_id=pattern.profile_id,
                target_name=pattern.profile_name,
                recommendation={
                    "action": "increase_overlap",
                    "current_issue": "Important context may be split across chunks",
                    "suggested_overlap_tokens": 128,
                },
                evidence=[f"query: {q}" for q in pattern.sample_queries],
                confidence=confidence,
                description=(
                    f"Profile '{pattern.profile_name}' 下有 {pattern.occurrence_count} 次"
                    f"'缺少关键信息'反馈，建议增加分块重叠以保留上下文"
                ),
            )

        elif category == IssueCategory.CITATION_ERROR:
            return OptimizationSuggestion(
                type=SuggestionType.UPDATE_BOILERPLATE,
                target_id=pattern.profile_id,
                target_name=pattern.profile_name,
                recommendation={
                    "action": "review_boilerplate_patterns",
                    "current_issue": "Noise content may not be properly removed",
                },
                evidence=[f"query: {q}" for q in pattern.sample_queries],
                confidence=confidence,
                description=(
                    f"Profile '{pattern.profile_name}' 下有 {pattern.occurrence_count} 次"
                    f"'引用错误'反馈，建议检查噪声模式配置"
                ),
            )

        elif category == IssueCategory.FORMAT:
            return OptimizationSuggestion(
                type=SuggestionType.ADJUST_HEADING_RULES,
                target_id=pattern.profile_id,
                target_name=pattern.profile_name,
                recommendation={
                    "action": "review_heading_rules",
                    "current_issue": "Document structure may not be properly detected",
                },
                evidence=[f"query: {q}" for q in pattern.sample_queries],
                confidence=confidence,
                description=(
                    f"Profile '{pattern.profile_name}' 下有 {pattern.occurrence_count} 次"
                    f"'格式问题'反馈，建议检查标题识别规则"
                ),
            )

        elif category == "general_negative":
            return OptimizationSuggestion(
                type=SuggestionType.ADD_TERM,
                target_id=pattern.profile_id,
                target_name=pattern.profile_name,
                recommendation={
                    "action": "review_terminology",
                    "current_issue": "Queries may contain domain terms not in dictionary",
                    "sample_queries": pattern.sample_queries,
                },
                evidence=[f"query: {q}" for q in pattern.sample_queries],
                confidence=confidence * 0.7,  # Lower confidence for general negative
                description=(
                    f"Profile '{pattern.profile_name}' 下有 {pattern.occurrence_count} 次"
                    f"负面反馈，建议检查领域词典是否缺少相关术语"
                ),
            )

        return None
