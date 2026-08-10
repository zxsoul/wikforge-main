"""Universal Parser 触发条件与编排辅助测试（任务 10.9）。

覆盖：
- ``is_no_profile_match`` / ``is_quality_below_threshold`` / ``should_run_universal_parser``
  的判定矩阵。
- ``run_universal_parser_and_persist_candidate`` 的正常路径、save_candidate 抛
  ``ValueError`` 时只警告不冒泡、以及 ``db is None`` 跳过持久化。

Validates: Requirements 16
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.parsers.base import ParsedDocument
from app.services.universal_parser_trigger import (
    TRIGGER_NO_PROFILE_MATCH,
    TRIGGER_QUALITY_BELOW_THRESHOLD,
    is_no_profile_match,
    is_quality_below_threshold,
    run_universal_parser_and_persist_candidate,
    should_run_universal_parser,
)


# ─── is_no_profile_match ────────────────────────────────────────────


class TestIsNoProfileMatch:
    """``profile_id is None`` 或 ``profile_name == 'generic-text'`` 都视为「没有匹配」。"""

    def test_none_id_and_name_returns_true(self):
        assert is_no_profile_match(None, None) is True

    def test_none_id_with_generic_name_returns_true(self):
        assert is_no_profile_match(None, "generic-text") is True

    def test_real_id_with_generic_name_returns_true(self):
        # 即使 ID 非空，但兜底 Profile 不算真正匹配。
        assert is_no_profile_match("00000000-0000-0000-0000-000000000001", "generic-text") is True

    def test_real_id_with_specific_name_returns_false(self):
        assert (
            is_no_profile_match(
                "00000000-0000-0000-0000-000000000001",
                "chinese-technical-spec",
            )
            is False
        )


# ─── is_quality_below_threshold ─────────────────────────────────────


class TestIsQualityBelowThreshold:
    """阈值默认从 settings 读取，``None`` quality_score 始终返回 False。"""

    def test_none_score_returns_false(self):
        # 评分阶段尚未运行 → 不能强行触发。
        assert is_quality_below_threshold(None, threshold=0.7) is False

    def test_score_below_explicit_threshold(self):
        assert is_quality_below_threshold(0.5, threshold=0.7) is True

    def test_score_equal_threshold_returns_false(self):
        # 严格小于：等于阈值不算「低于」。
        assert is_quality_below_threshold(0.7, threshold=0.7) is False

    def test_score_above_threshold_returns_false(self):
        assert is_quality_below_threshold(0.9, threshold=0.7) is False

    @patch("app.services.universal_parser_trigger.get_settings")
    def test_threshold_defaults_to_settings(self, mock_settings):
        mock_settings.return_value = SimpleNamespace(QUALITY_FALLBACK_THRESHOLD=0.8)
        # 0.75 < 0.8 → True；如果 threshold 默认值没读 settings 就会拿到 0.7（False）。
        assert is_quality_below_threshold(0.75) is True
        assert is_quality_below_threshold(0.85) is False


# ─── should_run_universal_parser ────────────────────────────────────


class TestShouldRunUniversalParser:
    """综合判定 + 触发原因列表的稳定顺序。"""

    def test_no_profile_match_only(self):
        ok, reasons = should_run_universal_parser(
            profile_id=None,
            profile_name="generic-text",
            quality_score=0.95,
            threshold=0.7,
        )
        assert ok is True
        assert reasons == [TRIGGER_NO_PROFILE_MATCH]

    def test_quality_below_threshold_only(self):
        ok, reasons = should_run_universal_parser(
            profile_id="00000000-0000-0000-0000-000000000001",
            profile_name="chinese-technical-spec",
            quality_score=0.5,
            threshold=0.7,
        )
        assert ok is True
        assert reasons == [TRIGGER_QUALITY_BELOW_THRESHOLD]

    def test_both_reasons_in_stable_order(self):
        ok, reasons = should_run_universal_parser(
            profile_id=None,
            profile_name=None,
            quality_score=0.4,
            threshold=0.7,
        )
        assert ok is True
        # no_profile_match 优先于 quality_below_threshold。
        assert reasons == [TRIGGER_NO_PROFILE_MATCH, TRIGGER_QUALITY_BELOW_THRESHOLD]

    def test_neither_returns_false(self):
        ok, reasons = should_run_universal_parser(
            profile_id="00000000-0000-0000-0000-000000000001",
            profile_name="chinese-technical-spec",
            quality_score=0.95,
            threshold=0.7,
        )
        assert ok is False
        assert reasons == []

    def test_no_quality_score_with_real_profile_returns_false(self):
        # 任务 11 上线前，管线传 quality_score=None；只要匹配到具体 Profile 就不触发。
        ok, reasons = should_run_universal_parser(
            profile_id="00000000-0000-0000-0000-000000000001",
            profile_name="chinese-technical-spec",
            quality_score=None,
        )
        assert ok is False
        assert reasons == []


# ─── run_universal_parser_and_persist_candidate ─────────────────────


def _make_processed_document(metadata: dict | None = None):
    """构造一个最小化的 ProcessedDocument-like 对象。"""
    from app.services.document_processor import ProcessedBlock, ProcessedDocument

    return ProcessedDocument(
        blocks=[ProcessedBlock(type="paragraph", text="hello", page_number=1)],
        metadata=metadata or {"file_type": "pdf"},
        markdown="hello",
    )


def _make_envelope() -> dict:
    return {
        "profile": {
            "name": "auto-generated-pdf-3p",
            "description": "Automatically generated by Universal Parser",
            "priority": 0,
            "enabled": False,
            "match_rules": {
                "filename_regex": [],
                "content_regex": [],
                "min_content_match_count": 1,
            },
            "heading_rules": [],
            "boilerplate": {
                "detection_mode": "both",
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
            "domain_dictionary_id": None,
        },
        "metadata": {
            "status": "pending_approval",
            "source": "universal_parser",
            "evidence": {
                "page_count": 3,
                "heading_count": 2,
                "table_count": 1,
                "boilerplate_candidates": 0,
                "avg_block_chars": 50.0,
            },
        },
    }


class TestRunUniversalParserAndPersistCandidate:
    """编排函数的三条路径：成功持久化、save_candidate 抛 ValueError、db=None。"""

    @pytest.mark.asyncio
    async def test_happy_path_persists_candidate(self):
        parsed_doc = ParsedDocument(blocks=[], metadata={"file_type": "pdf"})
        processed = _make_processed_document()
        envelope = _make_envelope()

        parser = MagicMock()
        parser.parse = AsyncMock(return_value=processed)
        parser.suggest_profile = AsyncMock(return_value=envelope)

        db = MagicMock()
        db.commit = AsyncMock()
        db.rollback = AsyncMock()

        saved_profile = SimpleNamespace(id="11111111-1111-1111-1111-111111111111")

        with patch(
            "app.services.profile_candidate_service.save_candidate",
            new=AsyncMock(return_value=saved_profile),
        ) as mock_save:
            outcome = await run_universal_parser_and_persist_candidate(
                parsed_doc, db=db, parser=parser
            )

        parser.parse.assert_awaited_once_with(parsed_doc)
        parser.suggest_profile.assert_awaited_once_with(processed)
        mock_save.assert_awaited_once_with(db, envelope)
        db.commit.assert_awaited_once()
        db.rollback.assert_not_awaited()

        assert outcome["candidate_profile_id"] == "11111111-1111-1111-1111-111111111111"
        assert outcome["trigger_reasons"] == []
        assert outcome["processed_document"] is not None
        # ``processed_document`` 应该是序列化后的 dict（asdict）。
        assert isinstance(outcome["processed_document"], dict)
        assert outcome["processed_document"]["markdown"] == "hello"

    @pytest.mark.asyncio
    async def test_save_candidate_value_error_logged_and_swallowed(self, caplog):
        parsed_doc = ParsedDocument(blocks=[], metadata={"file_type": "pdf"})
        processed = _make_processed_document()
        envelope = _make_envelope()

        parser = MagicMock()
        parser.parse = AsyncMock(return_value=processed)
        parser.suggest_profile = AsyncMock(return_value=envelope)

        db = MagicMock()
        db.commit = AsyncMock()
        db.rollback = AsyncMock()

        with patch(
            "app.services.profile_candidate_service.save_candidate",
            new=AsyncMock(side_effect=ValueError("envelope rejected: bad name")),
        ):
            with caplog.at_level("WARNING"):
                outcome = await run_universal_parser_and_persist_candidate(
                    parsed_doc, db=db, parser=parser
                )

        # 不抛、processed_document 仍正常返回，candidate_profile_id 为 None。
        assert outcome["candidate_profile_id"] is None
        assert outcome["processed_document"] is not None
        # 已经回滚事务，没有 commit。
        db.commit.assert_not_awaited()
        db.rollback.assert_awaited_once()
        # 日志里有兜底警告。
        assert any(
            "candidate envelope rejected" in record.message for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_db_none_skips_persistence(self):
        parsed_doc = ParsedDocument(blocks=[], metadata={"file_type": "pdf"})
        processed = _make_processed_document()
        envelope = _make_envelope()

        parser = MagicMock()
        parser.parse = AsyncMock(return_value=processed)
        parser.suggest_profile = AsyncMock(return_value=envelope)

        with patch(
            "app.services.profile_candidate_service.save_candidate",
            new=AsyncMock(return_value=SimpleNamespace(id="should-not-be-called")),
        ) as mock_save:
            outcome = await run_universal_parser_and_persist_candidate(
                parsed_doc, db=None, parser=parser
            )

        mock_save.assert_not_awaited()
        assert outcome["candidate_profile_id"] is None
        assert outcome["processed_document"] is not None
