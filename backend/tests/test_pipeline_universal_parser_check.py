"""Pipeline 集成测试：``universal_parser_check`` 任务（任务 10.9）。

覆盖：
- ``profile_id`` 为 None / ``profile_name == 'generic-text'`` 时触发 LLM 兜底
  并合并 metadata。
- ``profile_id`` 为真且 ``profile_name`` 是具体 Profile 时透传 ``match_result``。
- ``submit_pipeline`` 构建的 chain 在 ``profile_match`` 与 ``process_document``
  之间插入了 ``universal_parser_check`` 任务（顺序断言）。

Validates: Requirements 16
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.tasks.pipeline import universal_parser_check


def _call_task(task, *args, **kwargs):
    """Invoke a Celery task as a plain function whether or not Celery is installed.

    - When Celery is available the decorator wraps the function as a Task; the
      underlying callable lives on ``.run`` (or can be called via ``.apply``).
    - When Celery is NOT installed the decorator is a no-op and the function
      still expects ``self`` as the first positional argument.
    """
    if hasattr(task, "run") and callable(task.run):
        # Celery Task instance
        return task.run(*args, **kwargs)
    # No-op decorator path: function still takes ``self`` first.
    return task(MagicMock(), *args, **kwargs)


def _make_match_result(*, profile_id, profile_name, blocks=None, metadata=None) -> dict:
    return {
        "document_id": "00000000-0000-0000-0000-00000000abcd",
        "blocks": blocks
        or [
            {
                "type": "paragraph",
                "text": "raw block text",
                "bbox": None,
                "page_number": 1,
                "style": {},
            }
        ],
        "metadata": metadata or {"file_type": "pdf", "page_count": 3},
        "asset_count": 0,
        "profile_id": profile_id,
        "profile_name": profile_name,
    }


# ─── Trigger path ───────────────────────────────────────────────────


class TestUniversalParserCheckTriggers:
    """没有匹配到具体 Profile 时应运行 LLM 兜底并合并 metadata。"""

    def test_no_profile_match_invokes_orchestrator(self):
        match_result = _make_match_result(profile_id=None, profile_name="generic-text")

        # 编排函数返回伪造的处理结果：1 个新 block + universal_parser metadata。
        fake_outcome = {
            "processed_document": {
                "blocks": [
                    {
                        "type": "heading",
                        "text": "Cleaned heading",
                        "heading_level": 1,
                        "page_number": 1,
                        "is_noise": False,
                        "asset_ids": [],
                        "original_text": "",
                    },
                    {
                        "type": "paragraph",
                        "text": "Cleaned body",
                        "heading_level": 0,
                        "page_number": 1,
                        "is_noise": False,
                        "asset_ids": [],
                        "original_text": "",
                    },
                ],
                "metadata": {
                    "file_type": "pdf",
                    "page_count": 3,
                    "universal_parser": {
                        "successful_pages": [1, 2, 3],
                        "degraded_pages": [],
                        "failed_pages": [],
                        "page_errors": {},
                        "whole_doc_degraded": False,
                    },
                },
                "markdown": "# Cleaned heading\n\nCleaned body",
                "noise_removed_count": 0,
                "headings_detected": 1,
            },
            "candidate_profile_id": "22222222-2222-2222-2222-222222222222",
            "trigger_reasons": [],
        }

        with patch(
            "app.services.universal_parser_trigger.run_universal_parser_and_persist_candidate",
            new=AsyncMock(return_value=fake_outcome),
        ) as mock_run, patch(
            "app.tasks.pipeline._update_document_status"
        ):
            # 在测试环境里 ``app.core.database`` 不一定可导入（asyncpg 等驱动可能未装）。
            # 任务实现里对 DB 路径包了 try/except：第一次调用失败后会自动回退到
            # ``db=None`` 并重试，依旧能拿到 fake_outcome。
            result = _call_task(universal_parser_check, match_result)

        assert mock_run.await_count >= 1
        # 顶层 marker 都打上了。
        assert result["universal_parser_triggered"] is True
        assert result["universal_parser_trigger_reasons"] == ["no_profile_match"]
        assert result["candidate_profile_id"] == "22222222-2222-2222-2222-222222222222"

        # blocks 已经被 LLM 处理结果覆盖。
        assert len(result["blocks"]) == 2
        assert result["blocks"][0]["text"] == "Cleaned heading"
        assert result["blocks"][0]["style"]["heading_level"] == 1
        assert result["blocks"][1]["text"] == "Cleaned body"

        # metadata 中合并了 universal_parser envelope，原有 file_type 仍保留。
        assert result["metadata"]["file_type"] == "pdf"
        assert "universal_parser" in result["metadata"]
        assert result["metadata"]["universal_parser"]["successful_pages"] == [1, 2, 3]


# ─── Pass-through path ───────────────────────────────────────────────


class TestUniversalParserCheckPassthrough:
    """匹配到具体 Profile 时不调用编排函数，直接透传 match_result。"""

    def test_specific_profile_passes_through(self):
        match_result = _make_match_result(
            profile_id="11111111-1111-1111-1111-111111111111",
            profile_name="chinese-technical-spec",
        )

        with patch(
            "app.services.universal_parser_trigger.run_universal_parser_and_persist_candidate",
            new=AsyncMock(),
        ) as mock_run, patch(
            "app.tasks.pipeline._update_document_status"
        ):
            result = _call_task(universal_parser_check, match_result)

        mock_run.assert_not_awaited()
        # 透传时 match_result 应保持不变（同一个 dict 或等价拷贝）。
        assert result is match_result or result == match_result
        assert "universal_parser_triggered" not in result


# ─── submit_pipeline chain order ─────────────────────────────────────


class TestSubmitPipelineChainOrder:
    """submit_pipeline 应在 profile_match 与 process_document 之间插入 universal_parser_check。"""

    def test_chain_includes_universal_parser_check_in_order(self):
        # 跳过实际调度：用 patch 拦截 chain + apply_async，再检查传给 chain 的参数顺序。
        from app.tasks import pipeline as pipeline_module

        if not pipeline_module.CELERY_AVAILABLE:
            pytest.skip("Celery not available; submit_pipeline cannot run")

        captured_signatures = {}

        # 用 MagicMock 包装每个任务，以便 ``.s`` 调用能被识别。
        # 我们直接断言 chain 的实际位置参数。
        recorded_args = []

        def fake_chain(*args, **kwargs):
            recorded_args.extend(args)
            mock_chain = MagicMock()
            mock_chain.apply_async = MagicMock()
            return mock_chain

        with patch.object(pipeline_module, "chain", new=fake_chain):
            pipeline_module.submit_pipeline("00000000-0000-0000-0000-00000000abcd")

        # 期望 7 个 .s 签名：parse → profile_match → universal_parser_check
        # → process → chunk → embed → index。
        assert len(recorded_args) == 7

        # 每个签名都来自对应 task 的 ``.s``；celery Signature 对象自带 ``task`` 属性。
        task_names = []
        for sig in recorded_args:
            name = getattr(sig, "task", None) or getattr(sig, "name", None)
            task_names.append(name)

        # universal_parser_check 必须是第三个签名（索引 2），位于 profile_match 之后、
        # process_document 之前。
        assert "pipeline.universal_parser_check" in task_names
        assert task_names.index("pipeline.universal_parser_check") == 2
        # 顺序整体校验：profile_match (1) → universal_parser_check (2) → process_document (3)
        assert task_names[1] == "pipeline.profile_match"
        assert task_names[2] == "pipeline.universal_parser_check"
        assert task_names[3] == "pipeline.process_document"
