"""``SearchFeedback`` 持久化存储测试（任务 17.3）。

任务目标（需求 9.3 / 18.3）：

> 反馈数据应持久化存储到 PostgreSQL，包含查询文本、返回结果、用户、相关 profile、
> 时间戳等字段。

本测试覆盖三个层面，不依赖真实数据库：

1. **模型层契约**：``SearchFeedback`` SQLAlchemy 映射的字段、类型、外键、索引
   与 design.md 中 ``Feedback Loop`` 节定义一致。
2. **迁移脚本契约**：Alembic 初始迁移中存在 ``search_feedbacks`` 表，且字段
   与模型一致。
3. **服务层持久化路径**：``FeedbackService.create_feedback`` 把 query / results /
   user_id / related_profile_id 等字段正确写入 ORM 实例并触发 ``add → flush →
   refresh``。

Validates: Requirements 9.3
"""

from __future__ import annotations

import inspect
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.dialects.postgresql import JSONB, UUID

from app.models.search_feedback import SearchFeedback
from app.services.feedback_service import FeedbackService


# ─── 1. 模型层契约 ────────────────────────────────────────────────────


class TestSearchFeedbackModelSchema:
    """验证 ``SearchFeedback`` 表结构符合需求 9.3 与 design.md。"""

    def test_table_name_matches_migration(self):
        """表名应为 ``search_feedbacks``（与迁移脚本对齐）。"""
        assert SearchFeedback.__tablename__ == "search_feedbacks"

    def test_required_columns_exist(self):
        """需求 9.3 列出的字段必须全部映射到列。"""
        columns = {c.name for c in SearchFeedback.__table__.columns}
        # 主键 + 时间戳由 mixin 提供
        assert "id" in columns
        assert "created_at" in columns
        # 需求 9.3 显式要求的字段
        assert "user_id" in columns
        assert "query" in columns
        assert "returned_results" in columns
        assert "feedback_type" in columns
        assert "issue_category" in columns
        assert "comment" in columns
        assert "related_profile_id" in columns

    def test_query_is_not_null_text(self):
        """``query`` 必填，且使用 Text（PostgreSQL 不限长）。"""
        col = SearchFeedback.__table__.c.query
        assert col.nullable is False
        assert col.type.python_type is str

    def test_returned_results_uses_jsonb_and_is_not_null(self):
        """``returned_results`` 持久化为 JSONB（PostgreSQL 原生 JSON 索引支持）。"""
        col = SearchFeedback.__table__.c.returned_results
        assert col.nullable is False
        assert isinstance(col.type, JSONB)

    def test_feedback_type_is_short_string(self):
        """``feedback_type`` 长度 20，非空（thumbs_up / thumbs_down / issue）。"""
        col = SearchFeedback.__table__.c.feedback_type
        assert col.nullable is False
        assert col.type.length == 20

    def test_issue_category_is_optional_string(self):
        """``issue_category`` 仅在 ``feedback_type == issue`` 时填充，可空。"""
        col = SearchFeedback.__table__.c.issue_category
        assert col.nullable is True
        assert col.type.length == 30

    def test_comment_is_optional_text(self):
        """``comment`` 可空、Text 类型，长度由 API/服务层校验（≤500）。"""
        col = SearchFeedback.__table__.c.comment
        assert col.nullable is True
        assert col.type.python_type is str

    def test_user_id_is_uuid_with_set_null_fk(self):
        """``user_id`` 关联 ``users.id``，删除用户后保留反馈（SET NULL）。"""
        col = SearchFeedback.__table__.c.user_id
        assert col.nullable is True
        assert isinstance(col.type, UUID)

        fks = list(col.foreign_keys)
        assert len(fks) == 1, "user_id 应当且仅有一个外键约束"
        fk = fks[0]
        assert fk.column.table.name == "users"
        assert fk.column.name == "id"
        assert (fk.ondelete or "").upper() == "SET NULL"

    def test_related_profile_id_is_uuid_with_set_null_fk(self):
        """``related_profile_id`` 关联 ``document_profiles.id``，可空。"""
        col = SearchFeedback.__table__.c.related_profile_id
        assert col.nullable is True
        assert isinstance(col.type, UUID)

        fks = list(col.foreign_keys)
        assert len(fks) == 1, "related_profile_id 应当且仅有一个外键约束"
        fk = fks[0]
        assert fk.column.table.name == "document_profiles"
        assert fk.column.name == "id"
        assert (fk.ondelete or "").upper() == "SET NULL"

    def test_created_at_is_timezone_aware(self):
        """``created_at`` 必须带时区，便于按时间范围聚合分析（任务 17.4）。"""
        col = SearchFeedback.__table__.c.created_at
        assert col.nullable is False
        # SQLAlchemy DateTime(timezone=True) 暴露为 ``timezone=True``
        assert getattr(col.type, "timezone", False) is True

    def test_relationships_load_user_and_profile(self):
        """ORM 关系应能直达 ``user`` 与 ``related_profile`` 对象。"""
        mapper = sa_inspect(SearchFeedback)
        rel_names = {r.key for r in mapper.relationships}
        assert {"user", "related_profile"}.issubset(rel_names)

    def test_indexes_support_aggregation_queries(self):
        """复合索引覆盖「按 Profile + 时间」「按用户 + 时间」两类聚合查询。"""
        index_names = {ix.name for ix in SearchFeedback.__table__.indexes}
        assert "ix_search_feedback_profile_created" in index_names
        assert "ix_search_feedback_user_created" in index_names


# ─── 2. Alembic 迁移契约 ──────────────────────────────────────────────


class TestSearchFeedbackMigration:
    """初始迁移脚本应创建 ``search_feedbacks`` 表并匹配模型字段。"""

    @pytest.fixture(scope="class")
    def migration_source(self) -> str:
        path = (
            Path(__file__).resolve().parent.parent
            / "alembic"
            / "versions"
            / "20250101_0001_001_initial_schema.py"
        )
        assert path.exists(), f"未找到初始迁移脚本: {path}"
        return path.read_text(encoding="utf-8")

    def test_migration_creates_table(self, migration_source: str):
        """迁移中存在 ``op.create_table('search_feedbacks', ...)``。"""
        assert 'create_table(\n        "search_feedbacks"' in migration_source

    @pytest.mark.parametrize(
        "fragment",
        [
            'sa.Column("query", sa.Text()',
            'sa.Column("returned_results", postgresql.JSONB()',
            'sa.Column("feedback_type", sa.String(20)',
            'sa.Column("issue_category", sa.String(30)',
            'sa.Column("comment", sa.Text()',
            'sa.ForeignKey("users.id", ondelete="SET NULL")',
            'sa.ForeignKey("document_profiles.id", ondelete="SET NULL")',
        ],
    )
    def test_migration_contains_required_columns(
        self, migration_source: str, fragment: str
    ):
        """迁移字段定义与模型一致（query / results / FK 等）。"""
        assert fragment in migration_source, (
            f"迁移脚本缺少字段定义: {fragment}"
        )

    def test_migration_creates_aggregation_indexes(self, migration_source: str):
        """迁移创建按 profile / user + created_at 的复合索引。"""
        assert "ix_search_feedback_profile_created" in migration_source
        assert "ix_search_feedback_user_created" in migration_source


# ─── 3. 服务层持久化路径 ──────────────────────────────────────────────


def _make_db_session() -> AsyncMock:
    """构造 ``AsyncSession`` mock，仅暴露 add/flush/refresh，足够覆盖
    ``FeedbackService.create_feedback`` 的写入路径。"""
    db = AsyncMock()
    db.add = MagicMock()  # SQLAlchemy add 是同步方法
    db.flush = AsyncMock()
    db.refresh = AsyncMock()
    return db


class TestFeedbackPersistencePath:
    """``FeedbackService.create_feedback`` 必须把所有需求 9.3 字段写入 ORM 实例。"""

    async def test_persists_all_fields_via_session(self):
        """thumbs_down + 完整字段：每个值都映射到 ``SearchFeedback`` 实例。"""
        db = _make_db_session()
        service = FeedbackService(db)

        user_id = uuid.uuid4()
        profile_id = uuid.uuid4()

        feedback = await service.create_feedback(
            user_id=user_id,
            query="如何配置 SSO 登录？",
            returned_results=["chunk-1", "chunk-2", "chunk-3"],
            feedback_type="thumbs_down",
            comment="返回的内容跟 SSO 没关系",
            related_profile_id=str(profile_id),
        )

        # 1) 字段映射正确
        assert isinstance(feedback, SearchFeedback)
        assert feedback.user_id == user_id
        assert feedback.query == "如何配置 SSO 登录？"
        assert feedback.returned_results == ["chunk-1", "chunk-2", "chunk-3"]
        assert feedback.feedback_type == "thumbs_down"
        assert feedback.issue_category is None
        assert feedback.comment == "返回的内容跟 SSO 没关系"
        assert feedback.related_profile_id == profile_id  # 服务层完成 str → UUID 转换

        # 2) 持久化路径：add → flush → refresh
        db.add.assert_called_once_with(feedback)
        db.flush.assert_awaited_once()
        db.refresh.assert_awaited_once_with(feedback)

    async def test_persists_issue_with_category(self):
        """``feedback_type=issue`` 时 ``issue_category`` 必须被写入实例。"""
        db = _make_db_session()
        service = FeedbackService(db)

        feedback = await service.create_feedback(
            user_id=uuid.uuid4(),
            query="搜索结果里没有 2024 年的合同模板",
            returned_results=[],
            feedback_type="issue",
            issue_category="missing_info",
            comment="预期出现 2024 版",
        )

        assert feedback.feedback_type == "issue"
        assert feedback.issue_category == "missing_info"
        assert feedback.comment == "预期出现 2024 版"
        # 空列表也应原样持久化（returned_results 为 NOT NULL 列）
        assert feedback.returned_results == []
        db.add.assert_called_once()

    async def test_persists_without_optional_profile(self):
        """匿名 / 无关联 Profile 的反馈也能持久化。"""
        db = _make_db_session()
        service = FeedbackService(db)

        feedback = await service.create_feedback(
            user_id=uuid.uuid4(),
            query="点赞测试",
            returned_results=["chunk-x"],
            feedback_type="thumbs_up",
        )

        assert feedback.related_profile_id is None
        assert feedback.comment is None
        db.add.assert_called_once()
        db.flush.assert_awaited_once()

    async def test_invalid_feedback_type_does_not_persist(self):
        """非法 ``feedback_type`` 必须在写入前被拒绝。"""
        db = _make_db_session()
        service = FeedbackService(db)

        with pytest.raises(ValueError, match="feedback_type"):
            await service.create_feedback(
                user_id=uuid.uuid4(),
                query="测试",
                returned_results=[],
                feedback_type="not_a_real_type",
            )

        db.add.assert_not_called()
        db.flush.assert_not_awaited()

    async def test_issue_without_category_does_not_persist(self):
        """``issue`` 必须附带 ``issue_category``，否则不入库。"""
        db = _make_db_session()
        service = FeedbackService(db)

        with pytest.raises(ValueError, match="issue_category"):
            await service.create_feedback(
                user_id=uuid.uuid4(),
                query="测试",
                returned_results=[],
                feedback_type="issue",
            )

        db.add.assert_not_called()

    async def test_create_feedback_signature_covers_requirement_fields(self):
        """需求 9.3 列出的字段必须全部出现在公开签名中，作为契约固定。"""
        sig = inspect.signature(FeedbackService.create_feedback)
        # 排除 ``self``
        param_names = set(sig.parameters.keys()) - {"self"}
        required = {
            "user_id",
            "query",
            "returned_results",
            "feedback_type",
            "issue_category",
            "comment",
            "related_profile_id",
        }
        missing = required - param_names
        assert not missing, f"create_feedback 缺少字段参数: {missing}"
