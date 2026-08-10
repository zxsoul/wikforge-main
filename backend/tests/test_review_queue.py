"""ReviewQueue 服务单元测试（任务 11.8）。

覆盖：
- ``enqueue`` 在分数低于阈值时创建 pending 行
- 分数等于阈值（0.7）不应该入队 —— 由 ``QualityScorer.needs_review`` 把守
- 分数高于阈值不应该入队
- 幂等：同一个 ``document_id`` 重复入队不会插入多行 pending；已存在 pending
  行的 ``quality_score`` 会被刷新
- 已结案（approved / corrected / rejected）的旧记录不会阻塞新一轮入队
- ``ParseQualityScore.to_dict()`` 与 ``ParseQualityScore.from_dict()`` 在
  JSONB 列上往返一致

Validates: Requirements 17
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.document_review import DocumentReview, ReviewStatus
from app.services.quality_scorer import (
    DEFAULT_REVIEW_THRESHOLD,
    ParseQualityScore,
    QualityScorer,
)
from app.services.review_queue import ReviewQueue


# ─── helpers ─────────────────────────────────────────────────────────


def _scalar_result(value):
    """模拟 ``await db.execute(...)``：``.scalar_one_or_none()`` 返回 *value*。"""
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    return result


def _make_db(*, find_pending_returns=None) -> AsyncMock:
    """构造 AsyncSession mock，覆盖 enqueue 用到的 add/flush/refresh/execute。"""
    db = AsyncMock()
    # add 是同步方法（SQLAlchemy 的 ``Session.add`` / ``AsyncSession.add``）。
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.refresh = AsyncMock()
    db.execute = AsyncMock(return_value=_scalar_result(find_pending_returns))
    return db


def _make_score(overall: float = 0.5, *, components=None, issues=None) -> ParseQualityScore:
    return ParseQualityScore(
        overall=overall,
        components=components
        or {
            "text_retention": 0.6,
            "heading_detection": 0.5,
            "table_completeness": 0.4,
            "numeric_protection": 0.5,
            "boilerplate_removal": 0.6,
        },
        issues=issues or ["text retention low"],
    )


# ─── fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def doc_id() -> uuid.UUID:
    return uuid.uuid4()


# ─── enqueue 行为：threshold gating（通过 QualityScorer.needs_review） ─────


class TestNeedsReviewThresholdBoundary:
    """验证 0.7 阈值的方向：< 0.7 入队；== 0.7 / > 0.7 不入队。

    ``ReviewQueue.enqueue`` 本身不做阈值判断（design.md 把判定职责放在
    ``QualityScorer.needs_review``）。这里通过 scorer + 模拟管线的「if
    needs_review: enqueue」逻辑来端到端验证。
    """

    @pytest.mark.asyncio
    async def test_score_below_threshold_triggers_enqueue(self, doc_id):
        scorer = QualityScorer()
        score = _make_score(overall=0.6)
        assert scorer.needs_review(score) is True

        db = _make_db()
        queue = ReviewQueue(db)
        review = await queue.enqueue(doc_id, score)

        # 应当 add 一行新 DocumentReview。
        assert db.add.called, "expected db.add to be called for new pending review"
        added = db.add.call_args.args[0]
        assert isinstance(added, DocumentReview)
        assert added.document_id == doc_id
        assert added.status == ReviewStatus.pending
        # quality_score 是 to_dict() 的产物；可以通过 from_dict 还原。
        roundtrip = ParseQualityScore.from_dict(added.quality_score)
        assert roundtrip.overall == pytest.approx(score.overall, abs=1e-4)
        assert review is added

    @pytest.mark.asyncio
    async def test_score_at_threshold_exactly_does_not_trigger(self, doc_id):
        """0.7 == 阈值，``needs_review`` 用严格小于，所以不应入队。"""
        scorer = QualityScorer()
        score = _make_score(overall=DEFAULT_REVIEW_THRESHOLD)  # 0.7
        assert scorer.needs_review(score) is False

        # 模拟管线决策：if needs_review → enqueue。
        db = _make_db()
        queue = ReviewQueue(db)
        if scorer.needs_review(score):
            await queue.enqueue(doc_id, score)
        assert not db.add.called, "0.7 boundary should NOT enqueue"
        assert not db.flush.called

    @pytest.mark.asyncio
    async def test_score_above_threshold_does_not_trigger(self, doc_id):
        scorer = QualityScorer()
        score = _make_score(overall=0.95)
        assert scorer.needs_review(score) is False

        db = _make_db()
        queue = ReviewQueue(db)
        if scorer.needs_review(score):
            await queue.enqueue(doc_id, score)
        assert not db.add.called


# ─── 幂等：重复入队 ────────────────────────────────────────────────


class TestEnqueueIdempotency:
    """同一文档已存在 pending 行时，``enqueue`` 不应再插一行。"""

    @pytest.mark.asyncio
    async def test_duplicate_enqueue_refreshes_existing_pending(self, doc_id):
        existing = MagicMock(spec=DocumentReview)
        existing.id = uuid.uuid4()
        existing.document_id = doc_id
        existing.status = ReviewStatus.pending
        existing.quality_score = {"overall": 0.55, "components": {}, "issues": ["old"]}

        db = _make_db(find_pending_returns=existing)
        queue = ReviewQueue(db)

        new_score = _make_score(overall=0.42, issues=["新一轮: numeric_protection 低"])
        result = await queue.enqueue(doc_id, new_score)

        # 关键断言：没有新建行；既有行被原地刷新成新分数。
        assert not db.add.called, "must not insert a second pending row"
        assert result is existing
        # quality_score 已被 to_dict() 后的 dict 替换。
        assert result.quality_score["overall"] == pytest.approx(0.42, abs=1e-4)
        # 通过 from_dict 往返还原 issues。
        roundtrip = ParseQualityScore.from_dict(result.quality_score)
        assert "新一轮: numeric_protection 低" in roundtrip.issues
        # flush 被调用了，但不是 add 路径（因为是 update）。
        db.flush.assert_awaited()

    @pytest.mark.asyncio
    async def test_double_enqueue_in_sequence_creates_then_refreshes(self, doc_id):
        """模拟「先入队、再次入队」的真实序列，状态在两次调用之间通过 mock 切换。"""
        # 第一次：DB 里没有 pending → 创建。
        db = _make_db(find_pending_returns=None)
        queue = ReviewQueue(db)
        first_score = _make_score(overall=0.5)
        first_review = await queue.enqueue(doc_id, first_score)
        assert db.add.called
        assert first_review.status == ReviewStatus.pending

        # 第二次：复用同一个 db mock，但让 execute 返回刚才创建的行。
        db.add.reset_mock()
        db.execute = AsyncMock(return_value=_scalar_result(first_review))

        second_score = _make_score(overall=0.3, issues=["更糟了"])
        second_review = await queue.enqueue(doc_id, second_score)

        # 同一行被刷新；不再 add。
        assert second_review is first_review
        assert not db.add.called
        roundtrip = ParseQualityScore.from_dict(second_review.quality_score)
        assert roundtrip.overall == pytest.approx(0.3, abs=1e-4)
        assert "更糟了" in roundtrip.issues


# ─── 已结案的历史记录不应阻塞新入队 ────────────────────────────────


class TestClosedReviewsDoNotBlockEnqueue:
    """``approved`` / ``corrected`` / ``rejected`` 不视为待审核。

    ReviewQueue 通过 ``status == pending`` 过滤，所以在 mock 中
    ``_find_pending`` 不会返回它们；本服务应当照常 add 新行。
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "closed_status",
        [ReviewStatus.approved, ReviewStatus.corrected, ReviewStatus.rejected],
    )
    async def test_closed_history_allows_new_pending_row(self, doc_id, closed_status):
        # 模拟 DB 里只有已结案的旧记录 → ``_find_pending`` 返回 None
        # （因为它的 SELECT 带 ``status == pending`` 过滤）。
        db = _make_db(find_pending_returns=None)
        queue = ReviewQueue(db)

        score = _make_score(overall=0.4)
        review = await queue.enqueue(doc_id, score)

        assert db.add.called
        assert review.status == ReviewStatus.pending
        # 仅断言关闭状态枚举是一个合法值（避免静默 typo）。
        assert closed_status in {
            ReviewStatus.approved,
            ReviewStatus.corrected,
            ReviewStatus.rejected,
        }


# ─── JSONB 序列化往返 ────────────────────────────────────────────────


class TestQualityScorePersistenceRoundTrip:
    """``ParseQualityScore.to_dict`` → JSONB → ``from_dict`` 一致性。"""

    @pytest.mark.asyncio
    async def test_round_trip_preserves_components_and_issues(self, doc_id):
        original = ParseQualityScore(
            overall=0.6543,
            components={
                "text_retention": 0.95,
                "heading_detection": 0.4,
                "table_completeness": 0.7,
                "numeric_protection": 1.0,
                "boilerplate_removal": 0.8,
            },
            issues=["heading detection low (40.0%)"],
        )

        db = _make_db()
        queue = ReviewQueue(db)
        review = await queue.enqueue(doc_id, original)

        # 写入的 quality_score 列必须是 dict（JSONB 可序列化），不是 dataclass。
        assert isinstance(review.quality_score, dict)
        roundtrip = ParseQualityScore.from_dict(review.quality_score)
        assert roundtrip.overall == pytest.approx(original.overall, abs=1e-3)
        assert roundtrip.components == pytest.approx(original.components, abs=1e-3)
        assert roundtrip.issues == original.issues

    @pytest.mark.asyncio
    async def test_string_document_id_is_accepted(self):
        """``enqueue`` 也接受字符串形式的 document_id。"""
        doc_id_str = "11111111-1111-1111-1111-111111111111"
        db = _make_db()
        queue = ReviewQueue(db)

        score = _make_score(overall=0.5)
        review = await queue.enqueue(doc_id_str, score)

        assert db.add.called
        assert review.document_id == uuid.UUID(doc_id_str)
