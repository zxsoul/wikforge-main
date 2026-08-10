"""Review Queue service（任务 11.8）。

当 :class:`app.services.quality_scorer.QualityScorer` 算出的解析质量综合分
``score.overall`` 严格低于审核阈值（默认 ``0.7``，见
``app.services.quality_scorer.DEFAULT_REVIEW_THRESHOLD`` 与
``Settings.REVIEW_QUEUE_THRESHOLD``）时，应当把文档「入队」到人工审核
通道：写一行 ``DocumentReview``，``status='pending'``，``quality_score``
落到 JSONB 列里。

幂等性约定（来自任务 11.8 的需求）::

    enqueue(doc, score)
    enqueue(doc, score)   # 不应再插一行；已有 pending 行的 quality_score 被刷新

实现要点：

- 只把 *pending* 的旧记录视为「重复入队」。``approved`` / ``corrected`` /
  ``rejected`` 都属于历史快照（design.md 审核流：管理员通过 / 修正 / 驳回 是
  终态），它们被保留，新一轮低质量再写一条 *pending*。
- 不调 ``commit``：事务由调用方（FastAPI 路由 / 管线 worker）控制。本服务
  只负责 ``add`` / ``flush`` / ``refresh``，遵循其他业务服务（如
  ``profile_candidate_service.save_candidate``）的 SQLAlchemy 用法。
- ``ParseQualityScore`` 的序列化通过 :py:meth:`ParseQualityScore.to_dict`
  完成，反序列化（包括往返）由
  :py:meth:`ParseQualityScore.from_dict` 提供——前端 / 路由层从 JSONB 列
  读出来时会一致还原。

设计参考：design.md 「Quality Score + Review Queue（审核层）」一节。
"""

from __future__ import annotations

import logging
import uuid
from typing import Union

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document_review import DocumentReview, ReviewStatus
from app.services.quality_scorer import ParseQualityScore

logger = logging.getLogger(__name__)


def _coerce_uuid(document_id: Union[str, uuid.UUID]) -> uuid.UUID:
    """Normalise ``document_id`` to ``uuid.UUID``.

    Args:
        document_id: 字符串或 ``uuid.UUID``。

    Returns:
        ``uuid.UUID`` 实例。

    Raises:
        ValueError: 字符串不是合法 UUID（``uuid.UUID(str)`` 抛出）。
    """
    if isinstance(document_id, uuid.UUID):
        return document_id
    return uuid.UUID(str(document_id))


class ReviewQueue:
    """异步审核队列服务。

    与项目里其它服务（``UploadService`` / ``FeedbackService`` 等）一样，
    构造时接收一个 ``AsyncSession``，由调用方负责事务提交。
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    # ─── enqueue ──────────────────────────────────────────────────

    async def enqueue(
        self,
        document_id: Union[str, uuid.UUID],
        score: ParseQualityScore,
        extra_payload: dict | None = None,
    ) -> DocumentReview:
        """把文档入队到人工审核通道。

        幂等：同一个 ``document_id`` 在已存在 ``status='pending'`` 行时,
        不会再插入新行；改为把 ``quality_score`` 列刷新成新分数（便于
        管线重跑后管理员看到最新评估）。

        Args:
            document_id: 文档 UUID（``str`` 或 ``uuid.UUID``）。
            score: ``ParseQualityScore``，会通过 ``to_dict()`` 写入
                ``DocumentReview.quality_score`` JSONB 列。
            extra_payload: 可选附加字段，浅合并进 ``score.to_dict()`` 的输出。
                任务 11.10「文档并排预览 API」用它把
                ``ProcessedDocument.markdown`` 一并落到 JSONB（key:
                ``parsed_markdown``），让审核详情页无需再次跑解析就能拿到
                清洗后的 Markdown。当 ``extra_payload`` 与 ``score`` 字段重名
                时，``score`` 的字段优先（避免上游意外覆盖关键评分字段）。

        Returns:
            新建或被刷新的 ``DocumentReview`` 实例。

        Raises:
            ValueError: ``document_id`` 不是合法 UUID 字符串。
        """
        doc_uuid = _coerce_uuid(document_id)
        score_payload = score.to_dict()
        # 浅合并：``extra_payload`` 不能覆盖 ``overall/components/issues`` 等
        # 评分字段，仅补充新键（如 ``parsed_markdown``）。这一约束保护了
        # ``ParseQualityScore.from_dict`` 的往返一致性。
        if extra_payload:
            for key, value in extra_payload.items():
                if key not in score_payload:
                    score_payload[key] = value

        # 是否存在「未结案」的旧审核记录？
        existing = await self._find_pending(doc_uuid)
        if existing is not None:
            # 幂等：刷新 JSONB，保留 created_at / id 不变。
            existing.quality_score = score_payload
            await self.db.flush()
            await self.db.refresh(existing)
            logger.info(
                "ReviewQueue.enqueue: refreshed pending review %s for document %s "
                "(overall=%.4f)",
                existing.id,
                doc_uuid,
                score.overall,
            )
            return existing

        # 新建一条 pending 记录。
        review = DocumentReview(
            document_id=doc_uuid,
            quality_score=score_payload,
            status=ReviewStatus.pending,
        )
        self.db.add(review)
        await self.db.flush()
        await self.db.refresh(review)
        logger.info(
            "ReviewQueue.enqueue: created pending review %s for document %s "
            "(overall=%.4f, components=%s)",
            review.id,
            doc_uuid,
            score.overall,
            sorted(score.components.keys()),
        )
        return review

    # ─── helpers ──────────────────────────────────────────────────

    async def _find_pending(self, doc_uuid: uuid.UUID) -> DocumentReview | None:
        """查询给定文档当前的 *pending* 审核记录（最多一条）。

        约定数据库里同一文档至多只有一条 pending 行（由本服务的幂等性
        保证；如果由于历史数据出现多条 pending，``scalar_one_or_none``
        会抛 ``MultipleResultsFound``，这种情况应当通过运维迁移修复，
        而非在 enqueue 路径里悄悄合并）。
        """
        stmt = select(DocumentReview).where(
            DocumentReview.document_id == doc_uuid,
            DocumentReview.status == ReviewStatus.pending,
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
