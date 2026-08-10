"""Initial schema - all core models for the enterprise knowledge base.

Revision ID: 001
Revises: None
Create Date: 2025-01-01 00:00:00.000000+00:00

设计来源：
    .kiro/specs/enterprise-knowledge-base/design.md
        - PostgreSQL 数据模型（基础 ER 图）
        - 扩展数据模型（Profile / 词典 / 审核 / 反馈）

约定：
    * UUID 主键由数据库 ``gen_random_uuid()`` 生成（来自 ``pgcrypto`` 扩展，
      该扩展在升级阶段开头启用）。
    * 所有时间戳使用 ``timestamptz`` 并以 ``NOW()`` 作为服务端默认值。
    * jsonb 字段以适当的字面量给出默认值，与 SQLAlchemy 模型保持一致。
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# ──────────────────────────────────────────────────────────────────────────
# Enum types
# ──────────────────────────────────────────────────────────────────────────
DOCUMENT_STATUS = postgresql.ENUM(
    "pending",
    "parsing",
    "cleaning",
    "chunking",
    "embedding",
    "indexing",
    "completed",
    "failed",
    name="document_status",
    create_type=False,
)
RESOURCE_TYPE = postgresql.ENUM(
    "space", "document", name="resource_type", create_type=False
)
ACCESS_LEVEL = postgresql.ENUM(
    "invisible", "read", "write", name="access_level", create_type=False
)
REVIEW_STATUS = postgresql.ENUM(
    "pending",
    "approved",
    "corrected",
    "rejected",
    name="review_status",
    create_type=False,
)


def _uuid_pk_default():
    """Return the server-side default expression for UUID primary keys.

    ``gen_random_uuid()`` 由 ``pgcrypto`` 提供，避免对已废弃的
    ``uuid-ossp`` 扩展产生依赖。
    """
    return sa.text("gen_random_uuid()")


def upgrade() -> None:  # noqa: D401, C901 - migration script
    # ─── PostgreSQL extensions ────────────────────────────────────────
    op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')
    op.execute('CREATE EXTENSION IF NOT EXISTS "pg_trgm"')

    bind = op.get_bind()
    DOCUMENT_STATUS.create(bind, checkfirst=True)
    RESOURCE_TYPE.create(bind, checkfirst=True)
    ACCESS_LEVEL.create(bind, checkfirst=True)
    REVIEW_STATUS.create(bind, checkfirst=True)

    # ─── users ────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=_uuid_pk_default(),
        ),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(255), nullable=True),
        sa.Column("display_name", sa.String(100), nullable=True),
        sa.Column("oidc_provider", sa.String(64), nullable=True),
        sa.Column("oidc_subject", sa.String(255), nullable=True),
        sa.Column(
            "failed_login_count",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
        sa.Column("locked_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_users_email", "users", ["email"])
    # OIDC 身份的 partial unique index：仅在 provider+subject 同时非空时强制唯一
    op.create_index(
        "uq_users_oidc_provider_subject",
        "users",
        ["oidc_provider", "oidc_subject"],
        unique=True,
        postgresql_where=sa.text(
            "oidc_provider IS NOT NULL AND oidc_subject IS NOT NULL"
        ),
    )

    # ─── domain_dictionaries (must precede document_profiles FK) ──────
    op.create_table(
        "domain_dictionaries",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=_uuid_pk_default(),
        ),
        sa.Column("name", sa.String(100), nullable=False, unique=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "terms", postgresql.JSONB(), server_default="[]", nullable=False
        ),
        sa.Column(
            "synonyms", postgresql.JSONB(), server_default="[]", nullable=False
        ),
        sa.Column(
            "stop_words", postgresql.JSONB(), server_default="[]", nullable=False
        ),
        sa.Column("enabled", sa.Boolean(), server_default="true", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_domain_dictionaries_name", "domain_dictionaries", ["name"]
    )

    # ─── document_profiles ────────────────────────────────────────────
    op.create_table(
        "document_profiles",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=_uuid_pk_default(),
        ),
        sa.Column("name", sa.String(100), nullable=False, unique=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("priority", sa.Integer(), server_default="0", nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default="true", nullable=False),
        sa.Column(
            "match_rules", postgresql.JSONB(), server_default="{}", nullable=False
        ),
        sa.Column(
            "heading_rules",
            postgresql.JSONB(),
            server_default="[]",
            nullable=False,
        ),
        sa.Column(
            "boilerplate", postgresql.JSONB(), server_default="{}", nullable=False
        ),
        sa.Column(
            "tables", postgresql.JSONB(), server_default="{}", nullable=False
        ),
        sa.Column(
            "chunking", postgresql.JSONB(), server_default="{}", nullable=False
        ),
        sa.Column(
            "domain_dictionary_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("domain_dictionaries.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("version", sa.Integer(), server_default="1", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_document_profiles_name", "document_profiles", ["name"]
    )

    # ─── spaces ───────────────────────────────────────────────────────
    op.create_table(
        "spaces",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=_uuid_pk_default(),
        ),
        sa.Column("name", sa.String(50), nullable=False, unique=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "created_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_spaces_name", "spaces", ["name"])

    # ─── folders ──────────────────────────────────────────────────────
    op.create_table(
        "folders",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=_uuid_pk_default(),
        ),
        sa.Column(
            "space_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("spaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "parent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("folders.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("name", sa.String(50), nullable=False),
        sa.Column("depth", sa.Integer(), server_default="1", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "space_id", "parent_id", "name", name="uq_folder_sibling_name"
        ),
    )
    # 同空间下根目录（parent_id IS NULL）的名称唯一
    op.create_index(
        "uq_folder_sibling_name_root",
        "folders",
        ["space_id", "name"],
        unique=True,
        postgresql_where=sa.text("parent_id IS NULL"),
    )

    # ─── documents ────────────────────────────────────────────────────
    op.create_table(
        "documents",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=_uuid_pk_default(),
        ),
        sa.Column(
            "space_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("spaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "folder_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("folders.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("file_type", sa.String(20), nullable=False),
        sa.Column("file_size", sa.BigInteger(), nullable=False),
        sa.Column("storage_path", sa.String(1000), nullable=False),
        sa.Column(
            "status",
            DOCUMENT_STATUS,
            server_default="pending",
            nullable=False,
        ),
        sa.Column(
            "retry_count", sa.Integer(), server_default="0", nullable=False
        ),
        sa.Column("error_detail", sa.Text(), nullable=True),
        sa.Column("current_stage", sa.String(50), nullable=True),
        sa.Column(
            "progress_percent",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
        sa.Column(
            "matched_profile_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("document_profiles.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("quality_score", postgresql.JSONB(), nullable=True),
        sa.Column(
            "uploaded_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id"),
            nullable=False,
        ),
        sa.Column(
            "last_status_update",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_documents_status", "documents", ["status"])

    # ─── document_tags ────────────────────────────────────────────────
    op.create_table(
        "document_tags",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=_uuid_pk_default(),
        ),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tag_name", sa.String(30), nullable=False),
        sa.UniqueConstraint("document_id", "tag_name", name="uq_document_tag"),
    )
    op.create_index(
        "ix_document_tags_document_id", "document_tags", ["document_id"]
    )
    op.create_index("ix_document_tags_tag_name", "document_tags", ["tag_name"])

    # ─── permissions ──────────────────────────────────────────────────
    op.create_table(
        "permissions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=_uuid_pk_default(),
        ),
        sa.Column("resource_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("resource_type", RESOURCE_TYPE, nullable=False),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("access_level", ACCESS_LEVEL, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "resource_id",
            "resource_type",
            "user_id",
            name="uq_permission_resource_user",
        ),
    )
    op.create_index("ix_permissions_resource_id", "permissions", ["resource_id"])
    op.create_index("ix_permissions_user_id", "permissions", ["user_id"])
    op.create_index(
        "ix_permission_user_resource_type",
        "permissions",
        ["user_id", "resource_type"],
    )

    # ─── chat_sessions ────────────────────────────────────────────────
    op.create_table(
        "chat_sessions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=_uuid_pk_default(),
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "last_active_at", sa.DateTime(timezone=True), nullable=False
        ),
        sa.Column(
            "is_expired", sa.Boolean(), server_default="false", nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_chat_sessions_user_id", "chat_sessions", ["user_id"])

    # ─── chat_messages ────────────────────────────────────────────────
    op.create_table(
        "chat_messages",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=_uuid_pk_default(),
        ),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("citations", postgresql.JSONB(), nullable=True),
        sa.Column("sequence_number", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_chat_messages_session_id", "chat_messages", ["session_id"]
    )
    op.create_index(
        "ix_chat_messages_session_sequence",
        "chat_messages",
        ["session_id", "sequence_number"],
    )

    # ─── profile_versions ─────────────────────────────────────────────
    op.create_table(
        "profile_versions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=_uuid_pk_default(),
        ),
        sa.Column(
            "profile_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("document_profiles.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("snapshot", postgresql.JSONB(), nullable=False),
        sa.Column(
            "changed_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("change_note", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("profile_id", "version", name="uq_profile_version"),
    )
    op.create_index(
        "ix_profile_versions_profile_id", "profile_versions", ["profile_id"]
    )

    # ─── parser_plugin_configs ────────────────────────────────────────
    op.create_table(
        "parser_plugin_configs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=_uuid_pk_default(),
        ),
        sa.Column("name", sa.String(100), nullable=False, unique=True),
        sa.Column("import_path", sa.String(255), nullable=False),
        sa.Column(
            "supported_extensions",
            postgresql.JSONB(),
            server_default="[]",
            nullable=False,
        ),
        sa.Column("priority", sa.Integer(), server_default="0", nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default="true", nullable=False),
        sa.Column(
            "config", postgresql.JSONB(), server_default="{}", nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_parser_plugin_configs_name", "parser_plugin_configs", ["name"]
    )

    # ─── document_reviews ─────────────────────────────────────────────
    op.create_table(
        "document_reviews",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=_uuid_pk_default(),
        ),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("quality_score", postgresql.JSONB(), nullable=False),
        sa.Column(
            "status",
            REVIEW_STATUS,
            server_default="pending",
            nullable=False,
        ),
        sa.Column(
            "reviewer_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id"),
            nullable=True,
        ),
        sa.Column("reviewer_note", sa.Text(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_document_reviews_document_id", "document_reviews", ["document_id"]
    )

    # ─── search_feedbacks ─────────────────────────────────────────────
    op.create_table(
        "search_feedbacks",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=_uuid_pk_default(),
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("returned_results", postgresql.JSONB(), nullable=False),
        sa.Column("feedback_type", sa.String(20), nullable=False),
        sa.Column("issue_category", sa.String(30), nullable=True),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column(
            "related_profile_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("document_profiles.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_search_feedbacks_user_id", "search_feedbacks", ["user_id"]
    )
    op.create_index(
        "ix_search_feedback_profile_created",
        "search_feedbacks",
        ["related_profile_id", "created_at"],
    )
    op.create_index(
        "ix_search_feedback_user_created",
        "search_feedbacks",
        ["user_id", "created_at"],
    )


def downgrade() -> None:  # noqa: D401
    op.drop_table("search_feedbacks")
    op.drop_table("document_reviews")
    op.drop_table("parser_plugin_configs")
    op.drop_table("profile_versions")
    op.drop_table("chat_messages")
    op.drop_table("chat_sessions")
    op.drop_table("permissions")
    op.drop_table("document_tags")
    op.drop_table("documents")
    op.drop_table("folders")
    op.drop_table("spaces")
    op.drop_table("document_profiles")
    op.drop_table("domain_dictionaries")
    op.drop_table("users")

    bind = op.get_bind()
    REVIEW_STATUS.drop(bind, checkfirst=True)
    ACCESS_LEVEL.drop(bind, checkfirst=True)
    RESOURCE_TYPE.drop(bind, checkfirst=True)
    DOCUMENT_STATUS.drop(bind, checkfirst=True)

    op.execute('DROP EXTENSION IF EXISTS "pg_trgm"')
    op.execute('DROP EXTENSION IF EXISTS "pgcrypto"')
