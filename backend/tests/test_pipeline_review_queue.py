"""Pipeline 集成测试：``process_document`` 任务的审核队列入队（任务 11.8）。

覆盖：
- 当 ``QualityScorer.needs_review(score)`` 为真（``overall < 0.7``）时，
  ``process_document`` 调用 ``_enqueue_for_review_sync`` 把文档入队到
  ``DocumentReview`` 表，并把 ``review_enqueued`` 标记为 ``True``。
- 分数等于 0.7（边界）和高于 0.7 时不入队，``review_enqueued`` 为 ``False``。
- 当 ``ReviewQueue`` 入队失败时不阻塞下游，仅打 WARNING 日志。

Validates: Requirements 17
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.services.document_processor import ProcessedBlock, ProcessedDocument
from app.services.quality_scorer import ParseQualityScore
from app.tasks.pipeline import process_document


def _call_task(task, *args, **kwargs):
    """在 Celery 装饰路径与无 Celery 路径下都能调用任务函数。"""
    if hasattr(task, "run") and callable(task.run):
        return task.run(*args, **kwargs)
    return task(MagicMock(), *args, **kwargs)


def _make_match_result() -> dict:
    return {
        "document_id": "00000000-0000-0000-0000-00000000abcd",
        "blocks": [
            {
                "type": "paragraph",
                "text": "raw paragraph",
                "bbox": None,
                "page_number": 1,
                "style": {},
            }
        ],
        "metadata": {"file_type": "pdf", "page_count": 1},
        "asset_count": 0,
        "profile_id": None,
        "profile_name": "generic-text",
    }


def _make_processed_document() -> ProcessedDocument:
    return ProcessedDocument(
        blocks=[
            ProcessedBlock(
                type="paragraph",
                text="cleaned paragraph",
                page_number=1,
                is_noise=False,
            )
        ],
        metadata={"file_type": "pdf"},
        markdown="cleaned paragraph",
        noise_removed_count=0,
        headings_detected=0,
    )


def _patches(score: ParseQualityScore):
    """Helper: 一组共享的 patch，使 process_document 不真访问数据库 / DocumentProcessor。

    返回一个 contextmanager-like 列表，方便测试直接 ``with patch.multiple()`` 风格使用。
    """
    return score


# ─── 触发入队 ──────────────────────────────────────────────────────


class TestProcessDocumentEnqueuesOnLowScore:
    """``overall < 0.7`` → 必须入队，``review_enqueued`` 为 True。"""

    def test_low_score_triggers_review_queue_enqueue(self):
        match_result = _make_match_result()
        low_score = ParseQualityScore(
            overall=0.5,
            components={
                "text_retention": 0.5,
                "heading_detection": 0.5,
                "table_completeness": 0.5,
                "numeric_protection": 0.5,
                "boilerplate_removal": 0.5,
            },
            issues=["overall below threshold"],
        )

        processed = _make_processed_document()
        scorer = MagicMock()
        scorer.score.return_value = low_score
        scorer.needs_review.return_value = True

        with patch(
            "app.tasks.pipeline._update_document_status"
        ), patch(
            "app.tasks.pipeline._resolve_profile_for_processing",
            return_value=MagicMock(),
        ), patch(
            "app.services.document_processor.DocumentProcessor",
        ) as mock_processor_cls, patch(
            "app.services.quality_scorer.QualityScorer",
            return_value=scorer,
        ), patch(
            "app.tasks.pipeline._persist_document_quality_score"
        ) as mock_persist, patch(
            "app.tasks.pipeline._enqueue_for_review_sync",
            return_value=True,
        ) as mock_enqueue:
            mock_processor_cls.return_value.process.return_value = processed

            result = _call_task(process_document, match_result)

        # 1) 评分被持久化。
        assert mock_persist.called
        persisted_doc_id, persisted_score = mock_persist.call_args.args
        assert persisted_doc_id == match_result["document_id"]
        assert persisted_score["overall"] == pytest.approx(0.5, abs=1e-4)

        # 2) 触发入队，且参数是 ParseQualityScore（不是 dict），
        #    与 ``ReviewQueue.enqueue`` 的签名一致。
        assert mock_enqueue.called
        enq_doc_id, enq_score = mock_enqueue.call_args.args
        assert enq_doc_id == match_result["document_id"]
        assert isinstance(enq_score, ParseQualityScore)
        assert enq_score.overall == pytest.approx(0.5, abs=1e-4)

        # 3) 顶层 marker 透传给下游 chunk_document。
        assert result["review_enqueued"] is True
        assert result["quality_score"]["overall"] == pytest.approx(0.5, abs=1e-4)
        assert result["metadata"]["review_enqueued"] is True
        assert result["metadata"]["quality_score"]["overall"] == pytest.approx(0.5, abs=1e-4)


# ─── 不触发入队 ────────────────────────────────────────────────────


class TestProcessDocumentDoesNotEnqueue:
    """``overall == 0.7`` 边界 与 ``> 0.7`` 都不应触发入队。"""

    @pytest.mark.parametrize("overall", [0.7, 0.85, 0.99])
    def test_score_at_or_above_threshold_does_not_enqueue(self, overall):
        match_result = _make_match_result()
        score = ParseQualityScore(
            overall=overall,
            components={
                "text_retention": 0.9,
                "heading_detection": 0.9,
                "table_completeness": 0.9,
                "numeric_protection": 0.9,
                "boilerplate_removal": 0.9,
            },
            issues=[],
        )

        processed = _make_processed_document()
        scorer = MagicMock()
        scorer.score.return_value = score
        # 模拟真实 needs_review 行为：overall < 0.7 才入队。
        scorer.needs_review.return_value = overall < 0.7

        with patch(
            "app.tasks.pipeline._update_document_status"
        ), patch(
            "app.tasks.pipeline._resolve_profile_for_processing",
            return_value=MagicMock(),
        ), patch(
            "app.services.document_processor.DocumentProcessor",
        ) as mock_processor_cls, patch(
            "app.services.quality_scorer.QualityScorer",
            return_value=scorer,
        ), patch(
            "app.tasks.pipeline._persist_document_quality_score"
        ), patch(
            "app.tasks.pipeline._enqueue_for_review_sync",
            return_value=True,
        ) as mock_enqueue:
            mock_processor_cls.return_value.process.return_value = processed

            result = _call_task(process_document, match_result)

        mock_enqueue.assert_not_called()
        assert result["review_enqueued"] is False
        assert result["metadata"]["review_enqueued"] is False


# ─── 入队失败的兜底 ────────────────────────────────────────────────


class TestEnqueueFailureDoesNotBlockPipeline:
    """``ReviewQueue.enqueue`` 抛异常时管线照样返回结果，仅打 WARNING。"""

    def test_enqueue_failure_is_swallowed(self, caplog):
        match_result = _make_match_result()
        low_score = ParseQualityScore(
            overall=0.5,
            components={"text_retention": 0.5},
            issues=[],
        )
        processed = _make_processed_document()

        scorer = MagicMock()
        scorer.score.return_value = low_score
        scorer.needs_review.return_value = True

        with patch(
            "app.tasks.pipeline._update_document_status"
        ), patch(
            "app.tasks.pipeline._resolve_profile_for_processing",
            return_value=MagicMock(),
        ), patch(
            "app.services.document_processor.DocumentProcessor",
        ) as mock_processor_cls, patch(
            "app.services.quality_scorer.QualityScorer",
            return_value=scorer,
        ), patch(
            "app.tasks.pipeline._persist_document_quality_score"
        ), patch(
            "app.tasks.pipeline._enqueue_for_review_sync",
            side_effect=RuntimeError("DB unreachable"),
        ):
            mock_processor_cls.return_value.process.return_value = processed

            with caplog.at_level("WARNING"):
                result = _call_task(process_document, match_result)

        # 管线照常返回；仅 review_enqueued 为 False。
        assert result["review_enqueued"] is False
        assert result["quality_score"]["overall"] == pytest.approx(0.5, abs=1e-4)
        # 日志里有兜底警告。
        assert any(
            "failed to enqueue document" in record.message for record in caplog.records
        )
