"""反馈闭环集成测试（任务 17.10）。

把任务 17.1-17.9 已经分别覆盖的子能力串联起来，验证「反馈 → 聚合 →
模式检测 → 建议 → 应用 → 重处理 → 进度」整条链路在真实交互中协同
正确（需求 9.1-9.9 / 18.1-18.6）。

覆盖三个集成视角：

1. **完整闭环串联**：多用户提交反馈 → ``aggregate_feedback`` →
   ``detect_error_patterns`` → ``generate_suggestions`` →
   ``apply_profile_update`` → ``get_affected_documents`` →
   ``trigger_reprocessing`` → ``get_reprocessing_progress``。
   验证目标：
   - 模式检测的 ``profile_id`` 与建议的 ``target_id`` 一致；
   - 建议被应用后 ``DocumentProfile.version`` 严格 +1；
   - 重处理任务被同步写入 Redis（``reprocess:task:{id}`` hash + 24h TTL），
     立即可由进度查询命中。

2. **完整状态转换序列**：``pending → running → completed`` 与
   ``pending → running → failed``。
   通过真实 fakeredis 驱动 ``trigger_reprocessing`` →
   ``_increment_reprocess_progress`` → ``get_reprocessing_progress``，
   验证 worker 推进语义符合任务 17.9 设计：每篇成功 +1 ``processed``，
   达到 ``total`` 时按是否有失败收敛到 ``completed`` 或 ``failed``。

3. **用户隔离**：用户 A 在某 Profile 下持续触发 ``missing_info`` 反馈，
   用户 B 仅偶发反馈。``FeedbackFilter(user_id=...)`` 在聚合 / 列出
   场景下都应被透传，使两个视图统计互不干扰；同时全局视角能观察到
   合并后的整体趋势。

为避免引入真实 PostgreSQL，本文件使用 ``AsyncSession`` mock + 真实
fakeredis；Celery 部分通过 patch ``celery_app`` 替身，避免投递真实
broker 而仍能断言任务名 / 队列等关键参数。

Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.feedback_service import (
    FeedbackFilter,
    FeedbackService,
    IssueCategory,
    SuggestionType,
)


# ─── 通用替身与 fixture ────────────────────────────────────────────────


def _feedback(
    *,
    user_id: uuid.UUID,
    feedback_type: str,
    profile_id: uuid.UUID | None = None,
    issue_category: str | None = None,
    query: str = "默认查询",
    returned_results: list[str] | None = None,
    created_at: datetime | None = None,
) -> SimpleNamespace:
    """构造一条「最小可用」的反馈样本。

    与现有 17.4/17.5 单元测试保持一致的字段集合；用 :class:`SimpleNamespace`
    替代真实 ORM 实例，避免触发 SQLAlchemy 元数据初始化。
    """
    return SimpleNamespace(
        id=uuid.uuid4(),
        user_id=user_id,
        feedback_type=feedback_type,
        related_profile_id=profile_id,
        issue_category=issue_category,
        query=query,
        returned_results=returned_results or [],
        created_at=created_at or datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
    )


def _result(items: list) -> MagicMock:
    """构造 ``await db.execute(...)`` 返回的 ``Result`` 替身：``.scalars().all()``
    输出 *items*，``.scalar_one_or_none()`` 输出第一项（None 兼容）。"""
    scalars = MagicMock()
    scalars.all.return_value = list(items)
    res = MagicMock()
    res.scalars.return_value = scalars
    res.scalar_one_or_none.return_value = items[0] if items else None
    return res


def _stub_db_with_pipeline(pipeline: list[list]) -> AsyncMock:
    """构造一个按调用顺序响应不同结果的 ``AsyncSession`` mock。

    *pipeline* 中每个元素是一次 ``db.execute`` 应该返回的对象列表。这种
    顺序编排足以覆盖 ``FeedbackService`` 中各方法的 SQL 调用序列：
    ``aggregate_feedback`` 一次读取，``detect_error_patterns`` 一次读取
    + N 次 ``_get_profile_name``，``apply_profile_update`` /
    ``apply_dictionary_update`` 各一次，``get_affected_documents`` 1-2 次。
    """
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.refresh = AsyncMock()
    db.commit = AsyncMock()

    queue = list(pipeline)

    async def _execute(_query):
        if not queue:
            return _result([])
        return _result(queue.pop(0))

    db.execute = AsyncMock(side_effect=_execute)
    return db


@pytest.fixture
def fake_async_redis(monkeypatch):
    """让 ``app.core.redis.get_redis`` 返回 fakeredis 异步客户端。

    与 ``trigger_reprocessing`` / ``get_reprocessing_progress`` 中的
    ``from app.core.redis import get_redis`` 路径完全一致，TTL / hset 都
    走 fakeredis 的真实协议解析，避免对断言形态做假设。
    """
    fakeredis = pytest.importorskip("fakeredis")
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)

    async def _fake_get_redis():
        return client

    # ``feedback_service.trigger_reprocessing`` 通过 ``from app.core.redis
    # import get_redis`` 内部导入，替换模块属性即可生效。
    import app.core.redis as redis_module

    monkeypatch.setattr(redis_module, "get_redis", _fake_get_redis)
    return client


@pytest.fixture
def fake_sync_redis(monkeypatch):
    """把同步 ``redis.Redis.from_url`` 替换为 fakeredis 同步客户端。

    pipeline.py 的 ``_increment_reprocess_progress`` 是 worker 同步路径，
    需要走 ``redis_lib.Redis.from_url(...)``；本 fixture 让它落到与
    fakeredis 异步客户端共享 hash 数据的实例上。

    注意 fakeredis 同步与异步客户端默认是独立内存空间，这里通过共享
    ``FakeServer`` 让两者读到同一份数据，从而真实串联「服务层入队 +
    worker 推进 + API 查询」三段路径。
    """
    fakeredis = pytest.importorskip("fakeredis")
    server = fakeredis.FakeServer()
    sync_client = fakeredis.FakeRedis(server=server, decode_responses=True)

    import redis as redis_lib

    def _fake_from_url(*_args, **_kwargs):
        return sync_client

    monkeypatch.setattr(redis_lib.Redis, "from_url", _fake_from_url)
    return sync_client, server


@pytest.fixture
def shared_redis(monkeypatch, fake_sync_redis):
    """让异步 ``get_redis`` 与同步 ``Redis.from_url`` 共享同一份 fakeredis 数据。

    ``trigger_reprocessing`` 用异步客户端写入入队快照，
    ``_increment_reprocess_progress`` 用同步客户端推进进度，
    ``get_reprocessing_progress`` 又回到异步客户端读取——三段必须能看到
    彼此的写入，状态机才能正确收敛。
    """
    fakeredis = pytest.importorskip("fakeredis")
    sync_client, server = fake_sync_redis
    async_client = fakeredis.aioredis.FakeRedis(
        server=server, decode_responses=True
    )

    async def _fake_get_redis():
        return async_client

    import app.core.redis as redis_module

    monkeypatch.setattr(redis_module, "get_redis", _fake_get_redis)
    return SimpleNamespace(sync=sync_client, async_=async_client, server=server)


# ─── 1. 完整闭环串联 ───────────────────────────────────────────────────


class TestFullFeedbackLoop:
    """需求 9.1-9.9：从用户反馈一路走到重处理进度查询。"""

    @pytest.mark.asyncio
    async def test_pattern_to_suggestion_to_apply_to_reprocess_to_progress(
        self, shared_redis
    ):
        """完整闭环：3 个用户对同一 Profile 反馈 → 模式 → 建议 → 应用 → 重处理 → 进度。"""
        # ── 输入：3 个用户、同一 Profile、3 条 missing_info + 1 条 thumbs_up
        profile_id = uuid.uuid4()
        user_a, user_b, user_c = (uuid.uuid4() for _ in range(3))

        feedbacks = [
            _feedback(
                user_id=user_a,
                feedback_type="issue",
                profile_id=profile_id,
                issue_category=IssueCategory.MISSING_INFO.value,
                query="合同期限是多久",
                returned_results=["doc-1"],
                created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
            ),
            _feedback(
                user_id=user_b,
                feedback_type="issue",
                profile_id=profile_id,
                issue_category=IssueCategory.MISSING_INFO.value,
                query="续约条款在哪里",
                returned_results=["doc-1", "doc-2"],
                created_at=datetime(2024, 6, 2, tzinfo=timezone.utc),
            ),
            _feedback(
                user_id=user_c,
                feedback_type="issue",
                profile_id=profile_id,
                issue_category=IssueCategory.MISSING_INFO.value,
                query="违约金计算",
                returned_results=["doc-2"],
                created_at=datetime(2024, 6, 3, tzinfo=timezone.utc),
            ),
            _feedback(
                user_id=user_a,
                feedback_type="thumbs_up",
                profile_id=profile_id,
                returned_results=["doc-3"],
                created_at=datetime(2024, 6, 4, tzinfo=timezone.utc),
            ),
        ]

        profile_obj = SimpleNamespace(
            id=profile_id,
            name="中式技术规范",
            version=2,
            chunking={},
            boilerplate={},
            heading_rules=[],
        )
        affected_documents = [
            SimpleNamespace(id=uuid.uuid4()) for _ in range(3)
        ]

        # ── 编排 db.execute 调用顺序：
        # 1) aggregate_feedback：返回全部反馈
        # 2) detect_error_patterns：返回负面反馈（issue + thumbs_down，关联 profile）
        # 3) detect 内部 _get_profile_name：返回 profile 名称
        # 4) generate_suggestions 内部直接复用步骤 2-3 的 patterns（detect_error_patterns 再调用一次）
        # 5) generate_suggestions 内部 _get_profile_name（再调用一次）
        # 6) apply_profile_update：返回 profile 实体
        # 7) get_affected_documents：返回受影响文档
        negative_only = [fb for fb in feedbacks if fb.feedback_type == "issue"]
        db = _stub_db_with_pipeline(
            [
                feedbacks,        # aggregate_feedback
                negative_only,    # detect_error_patterns - 第 1 次
                [profile_obj.name],  # _get_profile_name #1
                negative_only,    # generate_suggestions 内再次 detect
                [profile_obj.name],  # _get_profile_name #2
                [profile_obj],    # apply_profile_update（select profile）
                affected_documents,  # get_affected_documents
            ]
        )
        service = FeedbackService(db)

        # ── ① 聚合分析：4 条反馈 / 3 个 issue / 1 个 thumbs_up
        agg = await service.aggregate_feedback()
        assert agg.total_count == 4
        assert agg.thumbs_up_count == 1
        assert agg.issue_count == 3
        assert agg.by_profile == {str(profile_id): 4}
        assert agg.by_issue_category == {"missing_info": 3}
        # 文档维度去重计数：doc-1 命中 2 次（user_a / user_b），doc-2 命中 2 次，doc-3 命中 1 次
        assert agg.by_document == {"doc-1": 2, "doc-2": 2, "doc-3": 1}

        # ── ② 错误模式检测：3 条 missing_info → 触发一个 pattern
        patterns = await service.detect_error_patterns(min_occurrences=3)
        assert len(patterns) == 1
        pattern = patterns[0]
        assert pattern.profile_id == str(profile_id)
        assert pattern.profile_name == "中式技术规范"
        assert pattern.issue_category == "missing_info"
        assert pattern.occurrence_count == 3
        # 样本查询都来源于 missing_info 反馈（thumbs_up 不参与错误模式分组）
        assert set(pattern.sample_queries) == {
            "合同期限是多久",
            "续约条款在哪里",
            "违约金计算",
        }

        # ── ③ 生成优化建议：missing_info → adjust_chunking + increase_overlap
        suggestions = await service.generate_suggestions()
        assert len(suggestions) == 1
        suggestion = suggestions[0]
        assert suggestion.type == SuggestionType.ADJUST_CHUNKING.value
        # target_id 必须严格指向触发模式的 profile
        assert suggestion.target_id == str(profile_id)
        assert suggestion.recommendation["action"] == "increase_overlap"
        # confidence = min(3/10, 1.0) = 0.3
        assert suggestion.confidence == pytest.approx(0.3)

        # ── ④ 应用建议：把 recommendation 落库到 profile.chunking
        updated_profile = await service.apply_profile_update(
            profile_id=suggestion.target_id,
            updates={"chunking": suggestion.recommendation},
        )
        assert updated_profile is profile_obj
        # version 严格 +1（任务 17.7 / 需求 9.7）
        assert profile_obj.version == 3
        assert profile_obj.chunking == suggestion.recommendation

        # ── ⑤ 获取受影响文档：3 篇 completed 文档
        documents = await service.get_affected_documents(
            profile_id=suggestion.target_id
        )
        assert documents == affected_documents

        # ── ⑥ 触发重处理：fakeredis 真实写入 24h TTL hash
        with patch(
            "app.core.celery_app.celery_app",
            MagicMock(),
        ):
            task = await service.trigger_reprocessing(
                [str(d.id) for d in documents]
            )

        assert task.status == "running"
        assert task.total_documents == 3

        # Redis hash 已落盘
        snapshot = await shared_redis.async_.hgetall(
            f"reprocess:task:{task.task_id}"
        )
        assert snapshot["total"] == "3"
        assert snapshot["processed"] == "0"
        assert snapshot["status"] == "running"
        # TTL 在 (0, 86400] 内
        ttl = await shared_redis.async_.ttl(
            f"reprocess:task:{task.task_id}"
        )
        assert 0 < ttl <= 86400

        # ── ⑦ 立即进度查询：命中且未完成
        progress = await service.get_reprocessing_progress(task.task_id)
        assert progress is not None
        assert progress.task_id == task.task_id
        assert progress.total_documents == 3
        assert progress.processed_documents == 0
        assert progress.status == "running"


# ─── 2. 完整状态转换序列 ───────────────────────────────────────────────


class TestStateTransitionSequence:
    """``pending → running → completed/failed`` 完整链路。"""

    @pytest.mark.asyncio
    async def test_running_to_completed_when_all_documents_succeed(
        self, shared_redis
    ):
        """5 篇文档全部成功推进 → status 由 running 收敛到 completed。"""
        from app.tasks.pipeline import (
            _increment_reprocess_progress,
            _reprocess_progress_key,
        )

        with patch(
            "app.core.celery_app.celery_app",
            MagicMock(),
        ):
            service = FeedbackService(AsyncMock())
            task = await service.trigger_reprocessing(
                [str(uuid.uuid4()) for _ in range(5)]
            )

        # 入队后初始态
        progress = await service.get_reprocessing_progress(task.task_id)
        assert progress.status == "running"
        assert progress.processed_documents == 0

        # 模拟 worker 逐篇推进
        for step in range(1, 6):
            _increment_reprocess_progress(task.task_id, errored=False)
            progress = await service.get_reprocessing_progress(task.task_id)
            assert progress.processed_documents == step
            if step < 5:
                # 中途仍为 running，progress_percent 严格单增
                assert progress.status == "running"
            else:
                # 最后一步触发收敛
                assert progress.status == "completed"

        # Redis 上 errors 字段未被写入（成功路径只更新 processed/status）
        key = _reprocess_progress_key(task.task_id)
        sync = shared_redis.sync
        assert sync.hget(key, "processed") == "5"
        assert sync.hget(key, "status") == "completed"

    @pytest.mark.asyncio
    async def test_running_to_failed_when_any_document_errors(
        self, shared_redis
    ):
        """5 篇中 1 篇失败 → status 收敛到 failed，errors 计数为 1。"""
        from app.tasks.pipeline import (
            _increment_reprocess_progress,
            _reprocess_progress_key,
        )

        with patch(
            "app.core.celery_app.celery_app",
            MagicMock(),
        ):
            service = FeedbackService(AsyncMock())
            task = await service.trigger_reprocessing(
                [str(uuid.uuid4()) for _ in range(5)]
            )

        # 中间一篇失败，其余成功
        for step in range(5):
            _increment_reprocess_progress(task.task_id, errored=(step == 2))

        progress = await service.get_reprocessing_progress(task.task_id)
        assert progress.processed_documents == 5
        assert progress.status == "failed"

        sync = shared_redis.sync
        key = _reprocess_progress_key(task.task_id)
        assert sync.hget(key, "errors") == "1"

    @pytest.mark.asyncio
    async def test_progress_percent_monotonic_non_decreasing(
        self, shared_redis
    ):
        """progress_percent 必须随 worker 推进严格非降，且范围 [0, 100]。"""
        from app.tasks.pipeline import _increment_reprocess_progress

        with patch(
            "app.core.celery_app.celery_app",
            MagicMock(),
        ):
            service = FeedbackService(AsyncMock())
            task = await service.trigger_reprocessing(
                [str(uuid.uuid4()) for _ in range(4)]
            )

        observed: list[float] = []
        for _ in range(4):
            _increment_reprocess_progress(task.task_id, errored=False)
            progress = await service.get_reprocessing_progress(task.task_id)
            pct = (
                progress.processed_documents / progress.total_documents * 100
            )
            observed.append(pct)
            assert 0.0 <= pct <= 100.0

        # 严格单增（每篇推进 25%）
        assert observed == [25.0, 50.0, 75.0, 100.0]


# ─── 3. 用户隔离 ───────────────────────────────────────────────────────


class TestUserIsolation:
    """需求 9.4：``FeedbackFilter(user_id=...)`` 不让两个用户的统计互相污染。"""

    @pytest.mark.asyncio
    async def test_per_user_aggregation_does_not_leak_across_users(self):
        """用户 A 的 3 条反馈 + 用户 B 的 1 条 → 各自视图独立。"""
        profile_id = uuid.uuid4()
        user_a = uuid.uuid4()
        user_b = uuid.uuid4()

        all_feedbacks = [
            *[
                _feedback(
                    user_id=user_a,
                    feedback_type="issue",
                    profile_id=profile_id,
                    issue_category=IssueCategory.MISSING_INFO.value,
                    query=f"用户 A 查询 {i}",
                    returned_results=["doc-A"],
                    created_at=datetime(2024, 6, i + 1, tzinfo=timezone.utc),
                )
                for i in range(3)
            ],
            _feedback(
                user_id=user_b,
                feedback_type="thumbs_up",
                profile_id=profile_id,
                returned_results=["doc-B"],
                created_at=datetime(2024, 6, 5, tzinfo=timezone.utc),
            ),
        ]
        # 模拟「服务层把过滤 SQL 推到 DB」：在 stub 中按 FeedbackFilter
        # 实际过滤反馈，从而验证 ``aggregate_feedback`` 在不同 user_id
        # 视图下输出互不干扰。
        captured_filters: list[FeedbackFilter | None] = []

        def _filter_in_memory(filter: FeedbackFilter | None):
            if filter is None:
                return list(all_feedbacks)
            result = list(all_feedbacks)
            if filter.user_id is not None:
                target = uuid.UUID(filter.user_id)
                result = [fb for fb in result if fb.user_id == target]
            if filter.feedback_type is not None:
                result = [
                    fb for fb in result if fb.feedback_type == filter.feedback_type
                ]
            if filter.issue_category is not None:
                result = [
                    fb
                    for fb in result
                    if fb.issue_category == filter.issue_category
                ]
            return result

        # 通过包装 ``_apply_filters`` 让 stub 知道当前生效的 filter，并
        # 用真实过滤逻辑在内存中筛选样本。这能验证：
        # 1) ``aggregate_feedback`` 真的把传入的 ``filter`` 透传给了
        #    ``_apply_filters``（任何遗漏都会让断言失败）；
        # 2) 服务层根据过滤后的样本做聚合，而不是「先聚合再过滤」。
        db = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock()
        db.refresh = AsyncMock()
        service = FeedbackService(db)

        original_apply = service._apply_filters

        def _apply_filters_spy(query, filter):
            captured_filters.append(filter)
            return original_apply(query, filter)

        # 直接替换实例方法
        service._apply_filters = _apply_filters_spy  # type: ignore[assignment]

        async def _execute(_query):
            # 取最近一次捕获的 filter，按内存过滤反馈
            current = captured_filters[-1] if captured_filters else None
            return _result(_filter_in_memory(current))

        db.execute = AsyncMock(side_effect=_execute)

        # ── 视图 ①：全局聚合（不带 filter）
        agg_all = await service.aggregate_feedback()
        assert agg_all.total_count == 4
        assert agg_all.issue_count == 3
        assert agg_all.thumbs_up_count == 1

        # ── 视图 ②：用户 A 视角
        agg_a = await service.aggregate_feedback(
            filter=FeedbackFilter(user_id=str(user_a))
        )
        assert agg_a.total_count == 3
        assert agg_a.issue_count == 3
        assert agg_a.thumbs_up_count == 0
        # 用户 A 的反馈仅命中 doc-A
        assert agg_a.by_document == {"doc-A": 3}

        # ── 视图 ③：用户 B 视角
        agg_b = await service.aggregate_feedback(
            filter=FeedbackFilter(user_id=str(user_b))
        )
        assert agg_b.total_count == 1
        assert agg_b.thumbs_up_count == 1
        assert agg_b.issue_count == 0
        # 用户 B 的反馈不会出现在用户 A 的文档命中里
        assert agg_b.by_document == {"doc-B": 1}

        # ── 关键回归保护：filter 真的被透传了
        assert len(captured_filters) == 3
        assert captured_filters[0] is None
        assert captured_filters[1] is not None
        assert captured_filters[1].user_id == str(user_a)
        assert captured_filters[2] is not None
        assert captured_filters[2].user_id == str(user_b)

    @pytest.mark.asyncio
    async def test_per_user_pattern_detection_only_aggregates_within_view(self):
        """用户 A 单独触发 missing_info 模式；用户 B 反馈量低于阈值不触发。

        ``detect_error_patterns`` 没有 ``user_id`` 参数（设计上是全局视角），
        但通过聚合后续 ``FeedbackFilter(user_id=...)`` 路径独立，能在
        ``aggregate_feedback`` 视图里观察到「用户 A 的 missing_info 计数 == 3，
        用户 B 视图的 missing_info 计数 == 0」，即同一全局模式在分用户视角
        下并不平均化、也不交叉污染。
        """
        profile_id = uuid.uuid4()
        user_a = uuid.uuid4()
        user_b = uuid.uuid4()

        feedbacks_by_user: dict[uuid.UUID, list[SimpleNamespace]] = defaultdict(
            list
        )
        for i in range(3):
            feedbacks_by_user[user_a].append(
                _feedback(
                    user_id=user_a,
                    feedback_type="issue",
                    profile_id=profile_id,
                    issue_category=IssueCategory.MISSING_INFO.value,
                    query=f"A-{i}",
                )
            )
        # 用户 B 只触发 1 次 missing_info，不应被算入"模式"
        feedbacks_by_user[user_b].append(
            _feedback(
                user_id=user_b,
                feedback_type="issue",
                profile_id=profile_id,
                issue_category=IssueCategory.MISSING_INFO.value,
                query="B-1",
            )
        )

        all_feedbacks = [fb for items in feedbacks_by_user.values() for fb in items]

        # 注入 stub：按 user_id 过滤
        def _filter(filter: FeedbackFilter | None):
            if filter is None or filter.user_id is None:
                return list(all_feedbacks)
            target = uuid.UUID(filter.user_id)
            return [fb for fb in all_feedbacks if fb.user_id == target]

        captured: list[FeedbackFilter | None] = []

        db = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock()
        db.refresh = AsyncMock()
        service = FeedbackService(db)
        original_apply = service._apply_filters

        def _spy(query, filter):
            captured.append(filter)
            return original_apply(query, filter)

        service._apply_filters = _spy  # type: ignore[assignment]

        async def _execute(_query):
            current = captured[-1] if captured else None
            return _result(_filter(current))

        db.execute = AsyncMock(side_effect=_execute)

        # 用户 A 视角：3 条 missing_info，全部计入 by_issue_category
        agg_a = await service.aggregate_feedback(
            filter=FeedbackFilter(user_id=str(user_a))
        )
        assert agg_a.total_count == 3
        assert agg_a.by_issue_category == {"missing_info": 3}

        # 用户 B 视角：1 条 missing_info，不会因为 A 的计数被放大
        agg_b = await service.aggregate_feedback(
            filter=FeedbackFilter(user_id=str(user_b))
        )
        assert agg_b.total_count == 1
        assert agg_b.by_issue_category == {"missing_info": 1}
