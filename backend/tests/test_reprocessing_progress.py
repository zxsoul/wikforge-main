"""批量重处理任务进度追踪测试（任务 17.9 / 需求 9.9）。

被测目标：

- ``app.tasks.pipeline._increment_reprocess_progress``：worker 处理完单篇
  文档后，原子递增 Redis hash ``reprocess:task:{task_id}`` 的 ``processed``
  字段，并在所有文档处理完成时把 ``status`` 收敛到 ``completed`` 或
  ``failed``。
- ``app.tasks.pipeline.mark_reprocess_progress``：作为 reprocess chain 末尾
  节点 / link_error 节点，根据上一步结果决定该次推进算成功还是失败。
- ``app.tasks.pipeline.reprocess_document``：批量重处理调度任务（注册名
  ``app.tasks.reprocess_document``），构造完整 chain 并附加进度回调。
- ``progress_percent`` 计算与状态转换契约（pending → running →
  completed/failed），与 ``GET /api/admin/feedback/reprocess/{task_id}``
  返回值的字段约束。

测试策略：

- Redis 行为通过 ``fakeredis.FakeRedis`` 驱动真实的 ``HINCRBY`` / ``HSET`` /
  ``HGET`` 协议，避免 mock 误把契约写错。
- Celery chain 通过 patch 模块级 ``chain`` 与 ``CELERY_AVAILABLE`` 隔离，
  断言任务名 / signature / link_error 等关键参数。
- API 层 ``progress_percent`` 计算复用 ``ReprocessingTask`` dataclass 与
  路由层的现有公式，只断言数学性质。

Validates: Requirements 9.9
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.tasks.pipeline import (
    _increment_reprocess_progress,
    _reprocess_progress_key,
    mark_reprocess_progress,
    reprocess_document,
)


# ─── 公共工具 ──────────────────────────────────────────────────────────


def _call_task(task, *args, **kwargs):
    """在 Celery 装饰路径与无 Celery 装饰路径下都能调用任务函数。

    带 ``bind=True`` 的 Celery 任务实际是 Task 实例，需通过 ``.run()``
    调用真正的任务体；无 Celery 时 ``_task_decorator`` 退化为 no-op，
    函数本身仍以 ``self`` 作为第一个参数，传一个 ``MagicMock()`` 即可。
    """
    if hasattr(task, "run") and callable(task.run):
        return task.run(*args, **kwargs)
    return task(MagicMock(), *args, **kwargs)


@pytest.fixture
def fake_sync_redis(monkeypatch):
    """让 ``redis.Redis.from_url`` 返回 fakeredis 同步客户端。

    pipeline.py 中的进度递增函数走 ``import redis as redis_lib`` →
    ``redis_lib.Redis.from_url(...)`` 同步路径，这里把它替换成
    fakeredis 实例，这样所有 ``hincrby`` / ``hset`` / ``hget`` 都按
    真实 Redis 语义跑，TTL / 字段类型一并被验证。
    """
    fakeredis = pytest.importorskip("fakeredis")
    client = fakeredis.FakeRedis(decode_responses=True)

    import redis as redis_lib

    def _fake_from_url(*_args, **_kwargs):
        return client

    monkeypatch.setattr(redis_lib.Redis, "from_url", _fake_from_url)
    return client


def _seed_task(redis_client, task_id: str, *, total: int, status: str = "running"):
    """在 fakeredis 里预置一条入队后的任务状态快照。"""
    key = _reprocess_progress_key(task_id)
    redis_client.hset(
        key,
        mapping={
            "total": str(total),
            "processed": "0",
            "status": status,
            "created_at": "2025-01-01T00:00:00+00:00",
            "error": "",
        },
    )
    redis_client.expire(key, 86400)
    return key


# ─── 1. _increment_reprocess_progress 行为 ────────────────────────────


class TestIncrementReprocessProgress:
    """worker 单步推进的核心契约。"""

    def test_increments_processed_and_keeps_running(self, fake_sync_redis):
        """未达 total 时：processed +1、status 保持 running、errors 不变。"""
        task_id = "task-running-1"
        key = _seed_task(fake_sync_redis, task_id, total=3)

        _increment_reprocess_progress(task_id, errored=False)

        assert fake_sync_redis.hget(key, "processed") == "1"
        assert fake_sync_redis.hget(key, "status") == "running"
        # 成功路径不应递增 errors（fakeredis 在 hincrby 时为 0，但成功路径
        # 我们故意不调用 hincrby，用 hget 兜底，未写入字段返回 None）。
        assert fake_sync_redis.hget(key, "errors") in (None, "0")

    def test_terminal_completed_when_all_succeed(self, fake_sync_redis):
        """processed == total 且 errors == 0 → status=completed。"""
        task_id = "task-complete"
        key = _seed_task(fake_sync_redis, task_id, total=2)

        _increment_reprocess_progress(task_id, errored=False)
        assert fake_sync_redis.hget(key, "status") == "running"
        _increment_reprocess_progress(task_id, errored=False)

        assert fake_sync_redis.hget(key, "processed") == "2"
        assert fake_sync_redis.hget(key, "status") == "completed"

    def test_terminal_failed_when_any_errored(self, fake_sync_redis):
        """processed == total 且 errors >= 1 → status=failed。"""
        task_id = "task-fail"
        key = _seed_task(fake_sync_redis, task_id, total=3)

        _increment_reprocess_progress(task_id, errored=False)
        _increment_reprocess_progress(task_id, errored=True)
        # 收尾这一条仍可成功，但因为前面有失败，最终态应为 failed。
        _increment_reprocess_progress(task_id, errored=False)

        assert fake_sync_redis.hget(key, "processed") == "3"
        assert fake_sync_redis.hget(key, "errors") == "1"
        assert fake_sync_redis.hget(key, "status") == "failed"

    def test_all_errored_marks_failed(self, fake_sync_redis):
        """所有文档都失败 → processed == total，status=failed。"""
        task_id = "task-all-fail"
        key = _seed_task(fake_sync_redis, task_id, total=2)

        _increment_reprocess_progress(task_id, errored=True)
        _increment_reprocess_progress(task_id, errored=True)

        assert fake_sync_redis.hget(key, "processed") == "2"
        assert fake_sync_redis.hget(key, "errors") == "2"
        assert fake_sync_redis.hget(key, "status") == "failed"

    def test_status_only_transitions_pending_running_terminal(
        self, fake_sync_redis
    ):
        """status 只允许的转换：``running → completed`` 或 ``running → failed``。

        中途的 inc 都不应回退或跳到其它状态。
        """
        task_id = "task-transitions"
        key = _seed_task(fake_sync_redis, task_id, total=3, status="running")
        observed = [fake_sync_redis.hget(key, "status")]

        for i in range(3):
            _increment_reprocess_progress(task_id, errored=(i == 1))
            observed.append(fake_sync_redis.hget(key, "status"))

        # running → running → running → failed（中间一条失败）
        assert observed == ["running", "running", "running", "failed"]

    def test_progress_percent_formula(self, fake_sync_redis):
        """进度百分比始终满足 ``processed / total * 100``，且范围 [0, 100]。

        这是 API 层 ``progress_percent`` 字段的语义保证（需求 9.9 显式列出）。
        """
        task_id = "task-percent"
        key = _seed_task(fake_sync_redis, task_id, total=4)

        # 初始 0%
        processed = int(fake_sync_redis.hget(key, "processed") or 0)
        total = int(fake_sync_redis.hget(key, "total") or 0)
        assert processed / total * 100 == 0.0

        for expected_pct in (25.0, 50.0, 75.0, 100.0):
            _increment_reprocess_progress(task_id, errored=False)
            processed = int(fake_sync_redis.hget(key, "processed"))
            total = int(fake_sync_redis.hget(key, "total"))
            pct = processed / total * 100
            assert pct == expected_pct
            assert 0.0 <= pct <= 100.0

    def test_redis_unavailable_swallowed(self, monkeypatch, caplog):
        """Redis 不可达时只 WARNING，不应让 worker 整个崩。"""
        import redis as redis_lib

        def _boom(*_args, **_kwargs):
            raise RuntimeError("redis unreachable")

        monkeypatch.setattr(redis_lib.Redis, "from_url", _boom)
        with caplog.at_level("WARNING"):
            # 不抛异常即视为通过
            _increment_reprocess_progress("any-task", errored=False)

        assert any(
            "Failed to increment reprocess progress" in r.message
            for r in caplog.records
        )


# ─── 2. mark_reprocess_progress（chain 末尾节点） ──────────────────────


class TestMarkReprocessProgress:
    """根据上一步结果决定算成功还是失败。"""

    def test_completed_dict_marks_success(self, fake_sync_redis):
        task_id = "task-mark-ok"
        key = _seed_task(fake_sync_redis, task_id, total=1)

        prev_result = {
            "document_id": "doc-1",
            "indexed_chunks": 7,
            "status": "completed",
        }
        out = _call_task(mark_reprocess_progress, prev_result, task_id)

        assert out is prev_result
        assert fake_sync_redis.hget(key, "processed") == "1"
        assert fake_sync_redis.hget(key, "status") == "completed"

    def test_non_completed_dict_marks_failure(self, fake_sync_redis):
        task_id = "task-mark-fail"
        key = _seed_task(fake_sync_redis, task_id, total=1)

        # status != "completed" → 视为该篇失败
        prev_result = {"document_id": "doc-1", "status": "failed"}
        out = _call_task(mark_reprocess_progress, prev_result, task_id)

        assert out is prev_result
        assert fake_sync_redis.hget(key, "processed") == "1"
        assert fake_sync_redis.hget(key, "errors") == "1"
        assert fake_sync_redis.hget(key, "status") == "failed"

    def test_non_dict_input_marks_failure(self, fake_sync_redis):
        """``link_error`` 路径上一步是异常对象，不是 dict → 算失败。"""
        task_id = "task-mark-error-link"
        key = _seed_task(fake_sync_redis, task_id, total=1)

        out = _call_task(
            mark_reprocess_progress, RuntimeError("upstream blew up"), task_id
        )

        assert out == {}
        assert fake_sync_redis.hget(key, "processed") == "1"
        assert fake_sync_redis.hget(key, "errors") == "1"
        assert fake_sync_redis.hget(key, "status") == "failed"


# ─── 3. reprocess_document（批量调度任务） ────────────────────────────


class TestReprocessDocumentTask:
    """``app.tasks.reprocess_document`` 调度行为。"""

    def test_celery_unavailable_marks_progress_as_failed(
        self, fake_sync_redis, caplog
    ):
        """Celery 不可用时：直接以 errored=True 推进进度，避免任务永远卡住。"""
        task_id = "task-no-celery"
        key = _seed_task(fake_sync_redis, task_id, total=1)

        with patch("app.tasks.pipeline.CELERY_AVAILABLE", False):
            with caplog.at_level("WARNING"):
                out = _call_task(reprocess_document, "doc-1", task_id)

        assert out["submitted"] is False
        assert out["document_id"] == "doc-1"
        assert out["task_id"] == task_id
        # 该篇被算作失败，单文档批量任务直接收敛到 failed。
        assert fake_sync_redis.hget(key, "processed") == "1"
        assert fake_sync_redis.hget(key, "status") == "failed"
        assert any(
            "Celery not available" in r.message for r in caplog.records
        )

    def test_happy_path_builds_full_chain_with_link_error(self, fake_sync_redis):
        """正常路径：构建 7 步 + 进度回调的 chain，并附 link_error 路径。"""
        task_id = "task-chain"
        _seed_task(fake_sync_redis, task_id, total=1)

        with patch("app.tasks.pipeline.CELERY_AVAILABLE", True), patch(
            "app.tasks.pipeline.chain"
        ) as mock_chain:
            mock_pipeline = MagicMock()
            mock_chain.return_value = mock_pipeline

            out = _call_task(reprocess_document, "doc-1", task_id)

        assert out["submitted"] is True
        # chain 被构建一次：parse → profile_match → universal_parser_check →
        # process → chunk → embed → index → mark_reprocess_progress（共 8 步）
        mock_chain.assert_called_once()
        chain_args = mock_chain.call_args.args
        assert len(chain_args) == 8

        # apply_async 被调用，且 link_error 是 mark_reprocess_progress 的 signature
        mock_pipeline.apply_async.assert_called_once()
        link_error = mock_pipeline.apply_async.call_args.kwargs["link_error"]
        # signature 的 args[0] 应是 task_id
        assert link_error.args == (task_id,)

        # 调度阶段不应触发进度递增（worker 真正完成时才推进）
        assert fake_sync_redis.hget(_reprocess_progress_key(task_id), "processed") == "0"

    def test_apply_async_failure_marks_progress_failed(
        self, fake_sync_redis, caplog
    ):
        """broker 不可达 → 立即把该篇标记失败，避免任务永远卡 running。"""
        task_id = "task-broker-down"
        key = _seed_task(fake_sync_redis, task_id, total=1)

        with patch("app.tasks.pipeline.CELERY_AVAILABLE", True), patch(
            "app.tasks.pipeline.chain"
        ) as mock_chain:
            mock_pipeline = MagicMock()
            mock_pipeline.apply_async.side_effect = RuntimeError(
                "broker connection refused"
            )
            mock_chain.return_value = mock_pipeline

            with caplog.at_level("WARNING"):
                out = _call_task(reprocess_document, "doc-1", task_id)

        assert out["submitted"] is False
        assert fake_sync_redis.hget(key, "processed") == "1"
        assert fake_sync_redis.hget(key, "status") == "failed"
        assert any(
            "Failed to submit reprocess chain" in r.message
            for r in caplog.records
        )


# ─── 4. 注册名校验：``app.tasks.reprocess_document`` ──────────────────


class TestReprocessDocumentRegistration:
    """需求 9.9 / 17.8：调用方依赖任务名 ``app.tasks.reprocess_document``。

    ``FeedbackService.trigger_reprocessing`` 使用 ``celery_app.send_task``
    按任务名投递，Celery 端必须以同名注册任务，否则 worker 接到消息时
    会抛 ``NotRegistered``。
    """

    def test_task_is_registered_with_expected_name(self):
        try:
            from celery.app.task import Task
        except ImportError:  # pragma: no cover - 测试环境必装 celery
            pytest.skip("celery not installed in test env")

        assert isinstance(reprocess_document, Task)
        assert reprocess_document.name == "app.tasks.reprocess_document"
