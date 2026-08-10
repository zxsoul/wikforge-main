"""Pipeline 测试：``submit_reprocess_from_markdown`` + ``_markdown_to_pipeline_blocks``
（任务 11.11）。

覆盖：
- ``_markdown_to_pipeline_blocks`` 的拆分行为：标题、段落、围栏代码块、
  空字符串、纯空白
- ``submit_reprocess_from_markdown`` 在 Celery 不可用时返回 False、不抛异常
- ``submit_reprocess_from_markdown`` 正常路径构建出
  ``cleanup → chunk → embed → index`` 的 Celery chain，并 ``apply_async``
- ``cleanup_document_indices`` 任务调用 ``IndexingService.delete_document_chunks``
- ``cleanup_document_indices`` 在 ``IndexingService`` 抛异常时仍透传 process_result

Validates: Requirements 17
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.tasks.pipeline import (
    _markdown_to_pipeline_blocks,
    cleanup_document_indices,
    submit_reprocess_from_markdown,
)


def _call_task(task, *args, **kwargs):
    """在 Celery 装饰路径与无 Celery 路径下都能调用任务函数。"""
    if hasattr(task, "run") and callable(task.run):
        return task.run(*args, **kwargs)
    return task(MagicMock(), *args, **kwargs)


# ─── _markdown_to_pipeline_blocks ──────────────────────────────────────


class TestMarkdownToPipelineBlocks:
    """Markdown → block 列表的最小拆分。"""

    def test_empty_string_returns_empty_list(self):
        assert _markdown_to_pipeline_blocks("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert _markdown_to_pipeline_blocks("   \n\n\t  ") == []

    def test_simple_paragraph(self):
        blocks = _markdown_to_pipeline_blocks("This is one paragraph.")
        assert len(blocks) == 1
        assert blocks[0]["type"] == "paragraph"
        assert blocks[0]["text"] == "This is one paragraph."
        assert blocks[0]["page_number"] == 1
        assert blocks[0]["style"] == {}

    def test_heading_levels(self):
        md = "# H1 title\n\n## H2 title\n\n###### H6 title"
        blocks = _markdown_to_pipeline_blocks(md)
        assert [b["type"] for b in blocks] == ["heading", "heading", "heading"]
        assert blocks[0]["text"] == "H1 title"
        assert blocks[0]["style"]["heading_level"] == 1
        assert blocks[1]["style"]["heading_level"] == 2
        assert blocks[2]["style"]["heading_level"] == 6

    def test_heading_then_paragraph(self):
        md = "# Section A\n\nFirst paragraph.\n\nSecond paragraph."
        blocks = _markdown_to_pipeline_blocks(md)
        assert len(blocks) == 3
        assert blocks[0]["type"] == "heading"
        assert blocks[0]["text"] == "Section A"
        assert blocks[1]["type"] == "paragraph"
        assert blocks[1]["text"] == "First paragraph."
        assert blocks[2]["type"] == "paragraph"
        assert blocks[2]["text"] == "Second paragraph."

    def test_fenced_code_block_kept_as_single_paragraph(self):
        md = "Before code.\n\n```python\nprint('hi')\nx = 1\n```\n\nAfter code."
        blocks = _markdown_to_pipeline_blocks(md)
        # before / fence / after
        assert len(blocks) == 3
        assert blocks[0]["text"] == "Before code."
        # 围栏代码块保持原貌（含围栏符号），整体作为一个 paragraph block。
        assert "```python" in blocks[1]["text"]
        assert "print('hi')" in blocks[1]["text"]
        assert blocks[1]["type"] == "paragraph"
        assert blocks[2]["text"] == "After code."

    def test_multiline_paragraph_kept_together(self):
        """段落内换行应视为一个 block（直到空行）。"""
        md = "Line one\nLine two\nLine three"
        blocks = _markdown_to_pipeline_blocks(md)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "paragraph"
        assert "Line one" in blocks[0]["text"]
        assert "Line three" in blocks[0]["text"]

    def test_multiple_blank_lines_do_not_create_empty_blocks(self):
        md = "Para A.\n\n\n\nPara B."
        blocks = _markdown_to_pipeline_blocks(md)
        assert len(blocks) == 2
        assert blocks[0]["text"] == "Para A."
        assert blocks[1]["text"] == "Para B."


# ─── cleanup_document_indices ──────────────────────────────────────────


class TestCleanupDocumentIndices:
    """链头任务：先清理旧切片再继续。"""

    def test_calls_indexing_service_delete(self):
        process_result = {
            "document_id": "00000000-0000-0000-0000-00000000abcd",
            "blocks": [],
            "metadata": {"reprocess_source": "manual_correction"},
        }

        with patch(
            "app.services.indexing_service.IndexingService"
        ) as mock_service_cls:
            mock_service = MagicMock()
            mock_service_cls.return_value = mock_service

            result = _call_task(cleanup_document_indices, process_result)

        mock_service.delete_document_chunks.assert_called_once_with(
            process_result["document_id"]
        )
        # 透传 process_result，让下游 chunk_document 直接消费。
        assert result is process_result

    def test_failure_does_not_raise(self, caplog):
        """``IndexingService`` 抛异常时只 WARNING，不让 chain 崩。"""
        process_result = {
            "document_id": "00000000-0000-0000-0000-00000000abcd",
            "blocks": [],
            "metadata": {},
        }

        with patch(
            "app.services.indexing_service.IndexingService"
        ) as mock_service_cls:
            mock_service = MagicMock()
            mock_service.delete_document_chunks.side_effect = RuntimeError(
                "qdrant unreachable"
            )
            mock_service_cls.return_value = mock_service

            with caplog.at_level("WARNING"):
                result = _call_task(cleanup_document_indices, process_result)

        assert result is process_result
        assert any(
            "Cleanup failed" in record.message for record in caplog.records
        )


# ─── submit_reprocess_from_markdown ────────────────────────────────────


class TestSubmitReprocessFromMarkdown:
    """高层 helper：从修正 Markdown 出发触发 reprocess chain。"""

    def test_celery_unavailable_returns_false(self, caplog):
        with patch("app.tasks.pipeline.CELERY_AVAILABLE", False):
            with caplog.at_level("WARNING"):
                ok = submit_reprocess_from_markdown(
                    "00000000-0000-0000-0000-00000000abcd",
                    "# Fixed\n\nbody",
                )
        assert ok is False
        assert any(
            "Celery not available" in record.message for record in caplog.records
        )

    def test_happy_path_builds_chain_and_dispatches(self):
        """正常路径：构建 cleanup → chunk → embed → index 的 chain 并 apply_async。"""
        # We patch the ``chain`` function imported at module level.
        with patch("app.tasks.pipeline.CELERY_AVAILABLE", True), patch(
            "app.tasks.pipeline.chain"
        ) as mock_chain:
            mock_pipeline = MagicMock()
            mock_chain.return_value = mock_pipeline

            ok = submit_reprocess_from_markdown(
                "00000000-0000-0000-0000-00000000abcd",
                "# 修正后标题\n\n这是修正过的正文。",
            )

        assert ok is True
        # chain 被构建一次，且 apply_async 被调用。
        mock_chain.assert_called_once()
        mock_pipeline.apply_async.assert_called_once()

        # chain 的第一个参数应当是 cleanup_document_indices 的 signature，且
        # 携带的 process_result 含解析出的 blocks。
        chain_args = mock_chain.call_args.args
        assert len(chain_args) == 4  # cleanup → chunk → embed → index
        first_signature = chain_args[0]
        # signature.args 是传给任务的位置参数；第一个是 process_result dict。
        process_result = first_signature.args[0]
        assert process_result["document_id"] == (
            "00000000-0000-0000-0000-00000000abcd"
        )
        assert process_result["metadata"]["reprocess_source"] == (
            "manual_correction"
        )
        # blocks 来自 _markdown_to_pipeline_blocks，第一个应为 heading。
        assert process_result["blocks"][0]["type"] == "heading"
        assert process_result["blocks"][0]["text"] == "修正后标题"

    def test_apply_async_failure_returns_false(self, caplog):
        """broker 不可达 → ``apply_async`` 抛异常 → 返回 False，不抛给上层。"""
        with patch("app.tasks.pipeline.CELERY_AVAILABLE", True), patch(
            "app.tasks.pipeline.chain"
        ) as mock_chain:
            mock_pipeline = MagicMock()
            mock_pipeline.apply_async.side_effect = RuntimeError(
                "broker connection refused"
            )
            mock_chain.return_value = mock_pipeline

            with caplog.at_level("WARNING"):
                ok = submit_reprocess_from_markdown(
                    "00000000-0000-0000-0000-00000000abcd",
                    "# Fixed\n\nbody",
                )

        assert ok is False
        assert any(
            "Failed to submit reprocess pipeline" in record.message
            for record in caplog.records
        )

    def test_empty_markdown_still_attempts_dispatch(self):
        """空 Markdown 也允许 dispatch（chain 内部会处理空 blocks 列表）。"""
        with patch("app.tasks.pipeline.CELERY_AVAILABLE", True), patch(
            "app.tasks.pipeline.chain"
        ) as mock_chain:
            mock_pipeline = MagicMock()
            mock_chain.return_value = mock_pipeline

            ok = submit_reprocess_from_markdown(
                "00000000-0000-0000-0000-00000000abcd", ""
            )

        # 即使 blocks 为空也应当走完调度路径——下游 chunk_document 已能
        # 正确处理空块列表，将索引清空。
        assert ok is True
        chain_args = mock_chain.call_args.args
        process_result = chain_args[0].args[0]
        assert process_result["blocks"] == []
