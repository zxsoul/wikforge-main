"""Pipeline 集成测试：管线状态更新（任务 12.8）。

覆盖：
- ``_update_document_status``：Redis 写入 ``stage``/``progress``/``updated_at``，
  并在 ``progress in {0, 100}`` 且 ``stage`` 为合法 ``DocumentStatus`` 时同步把
  状态推送到 PostgreSQL（``update_document_db_status``）。
- ``_update_document_status``：子步骤（如 ``profile_matching``）只写 Redis，
  不污染 PG 状态枚举。
- ``_update_document_status``：Redis 客户端抛异常时**不**向上传播，仅打 WARNING。
- ``_mark_document_failed``：把 PG ``status`` 置为 ``failed`` 并写 ``error_detail``。
- 管线各步骤（parse / profile_match / process / chunk / embed / index）在
  各自入口/出口正确触发状态更新，并最终由 ``index_chunks`` 把状态写为
  ``completed``。

Validates: Requirements 4
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.tasks.pipeline import (
    _mark_document_failed,
    _update_document_status,
    chunk_document,
    embed_chunks,
    index_chunks,
    parse_document,
    process_document,
    profile_match,
)


def _call_task(task, *args, **kwargs):
    """支持装饰过的 Celery Task 与 no-op 装饰路径下的统一调用。"""
    if hasattr(task, "run") and callable(task.run):
        return task.run(*args, **kwargs)
    return task(MagicMock(), *args, **kwargs)


# ─── _update_document_status ──────────────────────────────────────────


class TestUpdateDocumentStatus:
    """``_update_document_status`` 的核心契约。"""

    def test_redis_hset_with_stage_progress_updated_at(self):
        """Redis 应被写入 stage/progress/updated_at。"""
        mock_redis = MagicMock()

        with patch("redis.Redis") as mock_redis_class:
            mock_redis_class.from_url.return_value = mock_redis
            with patch(
                "app.services.indexing_service.update_document_db_status"
            ) as mock_pg:
                _update_document_status("doc-1", "embedding", 50)

        mock_redis_class.from_url.assert_called_once()
        mock_redis.hset.assert_called_once()
        # 检查 hset 的 mapping 关键字段
        _, kwargs = mock_redis.hset.call_args
        mapping = kwargs.get("mapping") or mock_redis.hset.call_args[0][1]
        assert mapping["stage"] == "embedding"
        assert mapping["progress"] == "50"
        assert "updated_at" in mapping
        # 中间进度（50）不更新 PG。
        mock_pg.assert_not_called()

    def test_progress_zero_updates_postgres_for_pipeline_stage(self):
        """步骤入口（progress=0）+ 合法 stage 应同时更新 PG。"""
        mock_redis = MagicMock()

        with patch("redis.Redis") as mock_redis_class:
            mock_redis_class.from_url.return_value = mock_redis
            with patch(
                "app.services.indexing_service.update_document_db_status"
            ) as mock_pg:
                _update_document_status("doc-1", "parsing", 0)

        mock_pg.assert_called_once_with(
            "doc-1",
            "parsing",
            current_stage="parsing",
            progress_percent=0,
        )

    def test_progress_hundred_updates_postgres_for_pipeline_stage(self):
        """步骤出口（progress=100）+ 合法 stage 应更新 PG。"""
        mock_redis = MagicMock()

        with patch("redis.Redis") as mock_redis_class:
            mock_redis_class.from_url.return_value = mock_redis
            with patch(
                "app.services.indexing_service.update_document_db_status"
            ) as mock_pg:
                _update_document_status("doc-1", "indexing", 100)

        mock_pg.assert_called_once_with(
            "doc-1",
            "indexing",
            current_stage="indexing",
            progress_percent=100,
        )

    def test_done_stage_maps_to_completed_in_postgres(self):
        """``stage='done'`` 应在 PG 侧落为 ``completed``（DocumentStatus 枚举）。"""
        mock_redis = MagicMock()

        with patch("redis.Redis") as mock_redis_class:
            mock_redis_class.from_url.return_value = mock_redis
            with patch(
                "app.services.indexing_service.update_document_db_status"
            ) as mock_pg:
                _update_document_status("doc-1", "done", 100)

        mock_pg.assert_called_once_with(
            "doc-1",
            "completed",
            current_stage="done",
            progress_percent=100,
        )

    def test_substage_does_not_touch_postgres(self):
        """``profile_matching`` 不在 DocumentStatus 枚举中，只写 Redis。"""
        mock_redis = MagicMock()

        with patch("redis.Redis") as mock_redis_class:
            mock_redis_class.from_url.return_value = mock_redis
            with patch(
                "app.services.indexing_service.update_document_db_status"
            ) as mock_pg:
                _update_document_status("doc-1", "profile_matching", 0)
                _update_document_status("doc-1", "profile_matching", 100)

        # Redis 两次都写
        assert mock_redis.hset.call_count == 2
        # PG 一次都不写
        mock_pg.assert_not_called()

    def test_redis_failure_does_not_propagate(self):
        """Redis 客户端抛异常时不应向上传播，仅记 WARNING。"""
        with patch("redis.Redis") as mock_redis_class:
            mock_redis_class.from_url.side_effect = ConnectionError("redis down")
            # PG 仍然应该被尝试调用。
            with patch(
                "app.services.indexing_service.update_document_db_status"
            ) as mock_pg:
                # 不抛异常即为通过。
                _update_document_status("doc-1", "parsing", 0)
                mock_pg.assert_called_once()

    def test_postgres_failure_does_not_propagate(self):
        """PG 写入异常时不应向上传播。"""
        mock_redis = MagicMock()
        with patch("redis.Redis") as mock_redis_class:
            mock_redis_class.from_url.return_value = mock_redis
            with patch(
                "app.services.indexing_service.update_document_db_status",
                side_effect=RuntimeError("pg down"),
            ):
                # 不抛异常即为通过。
                _update_document_status("doc-1", "parsing", 0)


# ─── _mark_document_failed ────────────────────────────────────────────


class TestMarkDocumentFailed:
    """失败路径必须把 PG ``status`` 标为 ``failed`` 并附带 ``error_detail``。"""

    def test_redis_records_failed_marker(self):
        mock_redis = MagicMock()

        with patch("redis.Redis") as mock_redis_class:
            mock_redis_class.from_url.return_value = mock_redis
            with patch(
                "app.services.indexing_service.update_document_db_status"
            ):
                _mark_document_failed("doc-1", "parsing", "boom")

        mock_redis.hset.assert_called_once()
        _, kwargs = mock_redis.hset.call_args
        mapping = kwargs.get("mapping") or mock_redis.hset.call_args[0][1]
        assert mapping["stage"] == "failed"
        assert mapping["progress"] == "0"
        assert mapping["error"] == "boom"
        assert mapping["failed_stage"] == "parsing"

    def test_postgres_status_set_to_failed_with_error_detail(self):
        mock_redis = MagicMock()

        with patch("redis.Redis") as mock_redis_class:
            mock_redis_class.from_url.return_value = mock_redis
            with patch(
                "app.services.indexing_service.update_document_db_status"
            ) as mock_pg:
                _mark_document_failed("doc-1", "embedding", "vector service oom")

        mock_pg.assert_called_once_with(
            "doc-1",
            "failed",
            current_stage="embedding",
            error_detail="vector service oom",
        )

    def test_redis_failure_does_not_block_postgres(self):
        """即使 Redis 挂了，PG 的 failed 标记仍要写入。"""
        with patch("redis.Redis") as mock_redis_class:
            mock_redis_class.from_url.side_effect = RuntimeError("redis down")
            with patch(
                "app.services.indexing_service.update_document_db_status"
            ) as mock_pg:
                _mark_document_failed("doc-1", "indexing", "boom")
                mock_pg.assert_called_once()


# ─── 管线步骤入口 / 出口的状态更新 ────────────────────────────────────


class _StatusRecorder:
    """收集 ``_update_document_status`` 调用序列，便于断言 stage 边界。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def __call__(self, document_id: str, stage: str, progress: int = 0) -> None:
        self.calls.append((document_id, stage, progress))

    def stages_with_progress(self, progress: int) -> list[str]:
        return [stage for _, stage, p in self.calls if p == progress]


class TestPipelineStepStatusBoundaries:
    """每个管线步骤都应在入口写 ``progress=0``、出口写 ``progress=100``。"""

    def test_parse_document_writes_zero_and_hundred(self):
        recorder = _StatusRecorder()
        with patch(
            "app.tasks.pipeline._update_document_status", side_effect=recorder
        ), patch(
            "app.tasks.pipeline._mark_document_failed"
        ), patch(
            "app.tasks.pipeline._get_document_info",
            return_value={"storage_path": "x.pdf", "file_type": "pdf"},
        ), patch(
            "app.tasks.pipeline._download_file_from_minio", return_value="/tmp/x.pdf"
        ), patch(
            "app.tasks.pipeline._ensure_default_parsers_registered"
        ), patch(
            "app.tasks.pipeline._get_mime_type", return_value="application/pdf"
        ), patch("os.path.exists", return_value=False):
            from app.services.parsers.base import ParsedDocument

            mock_parser = MagicMock()
            mock_parser.parse = MagicMock()

            async def _fake_parse(_path):
                return ParsedDocument(blocks=[], metadata={})

            mock_parser.parse = _fake_parse

            mock_registry = MagicMock()
            mock_registry.select.return_value = mock_parser

            with patch(
                "app.services.parsers.registry.get_parser_registry",
                return_value=mock_registry,
            ):
                _call_task(parse_document, "doc-1")

        assert "parsing" in recorder.stages_with_progress(0)
        assert "parsing" in recorder.stages_with_progress(100)

    def test_profile_match_writes_zero_and_hundred(self):
        recorder = _StatusRecorder()
        match_input = {
            "document_id": "doc-1",
            "blocks": [],
            "metadata": {},
        }

        with patch(
            "app.tasks.pipeline._update_document_status", side_effect=recorder
        ), patch(
            "app.tasks.pipeline._load_profiles_from_db", return_value=[]
        ), patch(
            "app.tasks.pipeline._get_document_info",
            return_value={"storage_path": "x.pdf", "file_type": "pdf"},
        ):
            _call_task(profile_match, match_input)

        assert "profile_matching" in recorder.stages_with_progress(0)
        assert "profile_matching" in recorder.stages_with_progress(100)

    def test_process_document_writes_zero_and_hundred(self):
        from app.services.document_processor import ProcessedDocument

        recorder = _StatusRecorder()
        match_result = {
            "document_id": "doc-1",
            "blocks": [
                {
                    "type": "paragraph",
                    "text": "hello",
                    "bbox": None,
                    "page_number": 1,
                    "style": {},
                }
            ],
            "metadata": {},
            "profile_id": None,
            "profile_name": "generic-text",
        }
        processed = ProcessedDocument(
            blocks=[], metadata={}, markdown="", noise_removed_count=0, headings_detected=0
        )

        scorer = MagicMock()
        scorer.score.return_value = MagicMock(
            overall=0.9, components={}, issues=[], to_dict=MagicMock(return_value={})
        )
        scorer.needs_review.return_value = False

        with patch(
            "app.tasks.pipeline._update_document_status", side_effect=recorder
        ), patch(
            "app.tasks.pipeline._resolve_profile_for_processing",
            return_value=MagicMock(),
        ), patch(
            "app.services.document_processor.DocumentProcessor"
        ) as mock_proc_cls, patch(
            "app.services.quality_scorer.QualityScorer", return_value=scorer
        ), patch(
            "app.tasks.pipeline._persist_document_quality_score"
        ):
            mock_proc_cls.return_value.process.return_value = processed
            _call_task(process_document, match_result)

        assert "cleaning" in recorder.stages_with_progress(0)
        assert "cleaning" in recorder.stages_with_progress(100)

    def test_chunk_document_writes_zero_and_hundred(self):
        recorder = _StatusRecorder()
        process_result = {
            "document_id": "doc-1",
            "blocks": [
                {
                    "type": "paragraph",
                    "text": "hello",
                    "page_number": 1,
                    "style": {},
                }
            ],
            "metadata": {},
            "profile_id": None,
        }

        with patch(
            "app.tasks.pipeline._update_document_status", side_effect=recorder
        ):
            _call_task(chunk_document, process_result)

        assert "chunking" in recorder.stages_with_progress(0)
        assert "chunking" in recorder.stages_with_progress(100)

    def test_embed_chunks_empty_chunks_writes_zero_and_hundred(self):
        recorder = _StatusRecorder()
        chunk_result = {
            "document_id": "doc-1",
            "chunks": [],
            "metadata": {},
            "profile_id": None,
        }

        with patch(
            "app.tasks.pipeline._update_document_status", side_effect=recorder
        ):
            _call_task(embed_chunks, chunk_result)

        assert "embedding" in recorder.stages_with_progress(0)
        assert "embedding" in recorder.stages_with_progress(100)

    def test_index_chunks_empty_marks_completed(self):
        """空 chunk 仍应进入「done/completed」终态，并调用 PG ``update_document_db_status``。"""
        recorder = _StatusRecorder()
        embed_result = {
            "document_id": "doc-1",
            "chunks": [],
            "embeddings": [],
            "metadata": {},
            "profile_id": None,
        }

        with patch(
            "app.tasks.pipeline._update_document_status", side_effect=recorder
        ), patch(
            "app.services.indexing_service.update_document_db_status"
        ) as mock_pg, patch(
            "app.services.indexing_service.update_pipeline_progress"
        ), patch(
            "app.core.qdrant.ensure_collection_exists"
        ), patch(
            "app.core.opensearch.ensure_index_exists"
        ):
            result = _call_task(index_chunks, embed_result)

        assert result["status"] == "completed"
        assert "indexing" in recorder.stages_with_progress(0)
        # 空 chunks 跳过中间步骤直接置 done/100
        assert "done" in recorder.stages_with_progress(100)
        # PG 显式被推到 completed
        mock_pg.assert_called_once_with("doc-1", "completed", "done", 100)

    def test_index_chunks_success_marks_completed(self):
        """有 chunks 时，``index_chunks`` 成功路径必须把 PG 标为 completed。"""
        recorder = _StatusRecorder()
        embed_result = {
            "document_id": "doc-1",
            "chunks": [
                {
                    "id": "c-1",
                    "text": "hello",
                    "chunk_index": 0,
                    "page_number": 1,
                }
            ],
            "embeddings": [
                {
                    "chunk_id": "c-1",
                    "dense_vector": [0.0] * 1024,
                    "sparse_indices": [],
                    "sparse_values": [],
                }
            ],
            "metadata": {},
            "profile_id": None,
        }

        mock_service = MagicMock()
        mock_service.index_chunks.return_value = {
            "qdrant_count": 1,
            "opensearch_count": 1,
        }

        with patch(
            "app.tasks.pipeline._update_document_status", side_effect=recorder
        ), patch(
            "app.tasks.pipeline._get_document_info",
            return_value={"storage_path": "x.pdf", "file_type": "pdf"},
        ), patch(
            "app.tasks.pipeline._get_document_space_id", return_value="space-1"
        ), patch(
            "app.services.indexing_service.IndexingService", return_value=mock_service
        ), patch(
            "app.services.indexing_service.update_document_db_status"
        ) as mock_pg, patch(
            "app.services.indexing_service.update_pipeline_progress"
        ), patch(
            "app.core.qdrant.ensure_collection_exists"
        ), patch(
            "app.core.opensearch.ensure_index_exists"
        ):
            result = _call_task(index_chunks, embed_result)

        assert result["status"] == "completed"
        assert result["indexed_chunks"] == 1
        # 入口 + done 出口都被记录
        assert "indexing" in recorder.stages_with_progress(0)
        assert "done" in recorder.stages_with_progress(100)
        mock_pg.assert_called_once_with("doc-1", "completed", "done", 100)


# ─── 失败路径 ─────────────────────────────────────────────────────────


class TestPipelineFailurePath:
    """步骤失败时应通过 ``_mark_document_failed`` 把 PG 标为 failed。"""

    def test_chunk_document_failure_marks_failed(self):
        """``chunk_document`` 在意外异常 + 重试用尽时，应调用 ``_mark_document_failed``。

        通过给 task 实例的 ``retry`` 方法注入 ``MaxRetriesExceededError`` 来模拟
        「最后一次重试」分支，验证此时 ``_mark_document_failed`` 会被调用。
        """
        from celery.exceptions import MaxRetriesExceededError

        bad_input = {
            "document_id": "doc-1",
            # 缺失 ``blocks`` 键 → KeyError → 走通用 except 分支。
            "metadata": {},
        }

        with patch("app.tasks.pipeline._update_document_status"), patch(
            "app.tasks.pipeline._mark_document_failed"
        ) as mock_fail, patch.object(
            chunk_document,
            "retry",
            side_effect=MaxRetriesExceededError(),
        ):
            with pytest.raises((MaxRetriesExceededError, KeyError)):
                _call_task(chunk_document, bad_input)

        mock_fail.assert_called_once()
        args, _ = mock_fail.call_args
        assert args[0] == "doc-1"
        assert args[1] == "chunking"
